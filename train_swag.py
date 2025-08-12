import os
import argparse
from datetime import datetime

import jax
import jax.numpy as jnp
import optax
from flax import nnx
import wandb
from tqdm import tqdm
import orbax.checkpoint as orbax
from typing import Type
import resnet
from dataloader_tfds import build_dataloader
import swag


def save_model_state(model: nnx.Module, path: str):
    state = nnx.state(model).to_pure_dict()
    # Save the parameters
    checkpointer = orbax.PyTreeCheckpointer()
    checkpointer.save(f"{path}/model", state)


def save_opt_state(optimizer: nnx.Optimizer, path: str):
    state = nnx.state(optimizer).to_pure_dict()
    checkpointer = orbax.PyTreeCheckpointer()
    checkpointer.save(f"{path}/opt", state)


def load_model_state(model_arch: Type[nnx.Module], path: str) -> nnx.Module:
    # create that model with abstract shapes
    model = nnx.eval_shape(lambda: model_arch(rngs=nnx.Rngs(0)))
    state = nnx.state(model)
    # Load the parameters
    checkpointer = orbax.PyTreeCheckpointer()
    state = checkpointer.restore(f"{path}/state", item=state)
    # update the model with the loaded state
    nnx.update(model, state)
    return model


def launch(args):
    model_key, swag_sample_key = jax.random.split(jax.random.key(args.seed))
    train_loader, val_loader = build_dataloader(args.batch_size)

    model_arch = resnet.__dict__[f"resnet{args.model_depth}"]
    model = model_arch(rngs=nnx.Rngs(model_key))
    graphdef, _, initial_batch_stats = nnx.split(model, nnx.Param, nnx.BatchStat)
    train_steps_per_epoch = 50000 // args.batch_size
    scheduler = optax.join_schedules(
        schedules=[
            optax.linear_schedule(
                init_value=0.0,
                end_value=args.optim_lr,
                transition_steps=args.warmup_epochs * train_steps_per_epoch,
            ),
            optax.cosine_decay_schedule(
                init_value=args.optim_lr,
                decay_steps=(args.start_swa_epoch - args.warmup_epochs)
                * train_steps_per_epoch,
                alpha=args.optim_swa_lr / args.optim_lr,
            ),
            optax.constant_schedule(
                value=args.optim_swa_lr,
            ),
        ],
        boundaries=[
            args.warmup_epochs * train_steps_per_epoch,
            args.start_swa_epoch * train_steps_per_epoch,
        ],
    )
    tx = optax.chain(
        optax.add_decayed_weights(weight_decay=args.optim_weight_decay),
        optax.sgd(scheduler, args.optim_momentum),
        swag.swag(
            freq=args.swag_freq,
            rank=args.swag_rank,
            start_step=args.start_swa_epoch * train_steps_per_epoch,
        ),
    )

    optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)
    metrics = nnx.MultiMetric(
        accuracy=nnx.metrics.Accuracy(),
        loss=nnx.metrics.Average("loss"),
    )

    def loss_fn(model: nnx.Module, batch):
        logits = model(batch["image"])
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits=logits, labels=batch["label"]
        ).mean()
        return loss, logits

    @nnx.jit
    def train_step(
        model: nnx.Module, optimizer: nnx.Optimizer, metrics: nnx.MultiMetric, batch
    ):
        """Train for a single step."""
        grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
        (loss, logits), grads = grad_fn(model, batch)
        metrics.update(loss=loss, logits=logits, labels=batch["label"])
        optimizer.update(grads)

    @nnx.jit
    def eval_step(model: nnx.Module, metrics: nnx.MultiMetric, batch):
        loss, logits = loss_fn(model, batch)
        metrics.update(loss=loss, logits=logits, labels=batch["label"])

    @nnx.jit
    def update_batch_stats_step(model: nnx.Module, batch):
        _ = model(batch["image"])

    def eval_swag(swag_state: optax.OptState, metrics: nnx.MultiMetric, key: jax.Array):
        weight_samples = swag.sample_swag(args.num_swag_samples, key, swag_state)
        swa_models = []
        for weight_sample in weight_samples:
            swa_model = nnx.merge(graphdef, weight_sample, initial_batch_stats)
            swa_model.train()
            train_epoch_loader = train_loader.take(train_steps_per_epoch)
            for train_batch in train_epoch_loader.as_numpy_iterator():
                update_batch_stats_step(swa_model, train_batch)
            swa_model.eval()
            swa_models.append(swa_model)
        for batch in val_loader.as_numpy_iterator():
            logits_list = []
            for swa_model in swa_models:
                _, logits = loss_fn(swa_model, batch)
            logits_list.append(logits)
            _logits = jnp.stack(logits_list)
            logprobs = nnx.log_softmax(_logits, axis=-1)
            ens_logprobs = nnx.logsumexp(logprobs, axis=0) - jnp.log(logprobs.shape[0])
            ens_nll = -jnp.sum(
                ens_logprobs * nnx.one_hot(batch["label"], ens_logprobs.shape[-1])
            ).mean()
            metrics.update(loss=ens_nll, logits=ens_logprobs, labels=batch["label"])

    with wandb.init(project="resnet-swag-cifar10", config=args) as run:
        for epoch in tqdm(range(args.optim_num_epochs)):
            model.train()
            train_epoch_loader = train_loader.take(train_steps_per_epoch)
            for batch in train_epoch_loader.as_numpy_iterator():
                train_step(model, optimizer, metrics, batch)
            # Log the train metrics.
            train_metrics_dict = {
                f"train/{metric}": value for metric, value in metrics.compute().items()
            }
            metrics.reset()  # Reset the metrics for the test set.

            # Compute the metrics on the test set after each training epoch.
            model.eval()
            for batch in val_loader.as_numpy_iterator():
                eval_step(model, metrics, batch)
            # Log the test metrics.
            test_metrics_dict = {
                f"test/{metric}": value for metric, value in metrics.compute().items()
            }
            metrics.reset()  # Reset the metrics for the next training epoch.
            run.log(
                {
                    **train_metrics_dict,
                    **test_metrics_dict,
                    "lr": scheduler(int(optimizer.step.value)).item(),
                    "steps": optimizer.step.value,
                },
                step=epoch,
            )
            if epoch >= args.start_swa_epoch and (
                epoch % args.eval_swag_freq == 0 or epoch == args.optim_num_epochs - 1
            ):
                swag_state = optimizer.opt_state[-1]
                swag_sample_subkey = jax.random.fold_in(swag_sample_key, swag_state.n)
                eval_swag(swag_state, metrics, swag_sample_subkey)
                swag_test_metrics_dict = {
                    **{
                        f"test/swag_{metric}": value
                        for metric, value in metrics.compute().items()
                    },
                    "test/num_swa_iterates": swag_state.n.item(),
                }
                metrics.reset()
                run.log({**swag_test_metrics_dict}, step=epoch)

            save_model_state(model, os.path.abspath(f"{args.save_dir}/{epoch+1:03d}"))
            save_opt_state(optimizer, os.path.abspath(f"{args.save_dir}/{epoch+1:03d}"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_depth", default=32, type=int, choices=[20, 32, 44, 56, 110]
    )
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--seed", default=42, type=int)

    parser.add_argument("--optim_lr", default=0.1, type=float)
    parser.add_argument("--optim_swa_lr", default=0.01, type=float)
    parser.add_argument("--optim_momentum", default=0.9, type=float)
    parser.add_argument("--optim_weight_decay", default=1e-4, type=float)
    parser.add_argument("--optim_num_epochs", default=1000, type=int)
    parser.add_argument("--warmup_epochs", default=10, type=int)
    parser.add_argument("--start_swa_epoch", default=800, type=int)
    parser.add_argument("--swag_freq", default=1, type=int)
    parser.add_argument("--swag_rank", default=10, type=int)
    parser.add_argument("--eval_swag_freq", default=10, type=int)
    parser.add_argument("--num_swag_samples", default=5, type=int)

    parser.add_argument("--save_dir", default="./checkpoint/swag", type=str)

    args = parser.parse_args()
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_dir = os.path.join(args.save_dir, now)
    args.save_dir = save_dir
    launch(args)


if __name__ == "__main__":
    main()
