import os
import argparse
from datetime import datetime, timezone, timedelta

import jax
import jax.numpy as jnp
import optax
from flax import nnx
import wandb
from tqdm import tqdm
import orbax.checkpoint as orbax
from typing import Optional
import models.resnet as resnet
from dataloader_tfds import build_dataloader
import swag
from eval import evaluate_nll, evaluate_ece


def create_checkpoint_manager(
    save_dir: str, max_to_keep: int = 5
) -> orbax.CheckpointManager:
    """Create a checkpoint manager with automatic cleanup."""
    options = orbax.CheckpointManagerOptions(
        max_to_keep=max_to_keep,
        create=True,
    )
    return orbax.CheckpointManager(save_dir, orbax.PyTreeCheckpointer(), options)


def save_checkpoint(
    checkpoint_manager: orbax.CheckpointManager,
    model: nnx.Module,
    optimizer: nnx.Optimizer,
    step: int,
    metrics: Optional[dict] = None,
):
    """Save checkpoint with model, optimizer state, and optional metrics."""
    state = {
        "model": nnx.state(model),
        "optimizer": nnx.state(optimizer),
        "step": step,
    }
    if metrics:
        state["metrics"] = metrics

    checkpoint_manager.save(step, state)


def load_checkpoint(
    model: nnx.Module,
    optimizer: nnx.Optimizer,
    checkpoint_manager: orbax.CheckpointManager,
    step: Optional[int] = None,
) -> int:
    """Load checkpoint into existing model and optimizer objects and return step."""
    # Get the latest step if not specified
    if step is None:
        step = checkpoint_manager.latest_step()
        if step is None:
            raise ValueError("No checkpoint found")

    # Load the checkpoint
    restored_state = checkpoint_manager.restore(step)

    # Update model and optimizer with loaded state
    nnx.update(model, restored_state["model"])
    nnx.update(optimizer, restored_state["optimizer"])

    return restored_state["step"]


def save_swag_state(
    swag_state: swag.SWAGState,
    save_path: str,
    model_id: Optional[str] = None,
):
    """Save SWAG state separately for inference."""
    if model_id:
        save_path = os.path.join(save_path, f"model_{model_id}")

    checkpoint_manager = orbax.CheckpointManager(
        save_path,
        orbax.PyTreeCheckpointer(),
        orbax.CheckpointManagerOptions(max_to_keep=1, create=True),
    )
    checkpoint_manager.save(0, {"swag_state": swag_state})
    print(f"SWAG state saved to {save_path}")


def load_swag_state(
    load_path: str,
) -> swag.SWAGState:  # Change return type annotation
    """Load SWAG state for inference."""
    checkpoint_manager = orbax.CheckpointManager(
        load_path,
        orbax.PyTreeCheckpointer(),
        orbax.CheckpointManagerOptions(max_to_keep=1, create=False),
    )
    restored_state = checkpoint_manager.restore(0)
    # Convert to proper SWAGState object
    swag_state_dict = restored_state["swag_state"]
    return swag.SWAGState(**swag_state_dict)


def launch(args):
    model_key, swag_sample_key = jax.random.split(jax.random.key(args.seed))
    train_loader, val_loader, train_steps_per_epoch = build_dataloader(
        args.ds_name, args.batch_size, args.seed
    )

    model_arch = resnet.__dict__[f"resnet{args.model_depth}"]
    model = model_arch(
        norm_type=args.norm_type, num_classes=args.num_classes, rngs=nnx.Rngs(model_key)
    )
    graphdef, _, initial_batch_stats = nnx.split(model, nnx.Param, nnx.BatchStat)

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
        ece=nnx.metrics.Average("ece"),
    )

    # Create checkpoint manager
    checkpoint_manager = create_checkpoint_manager(
        args.save_dir, max_to_keep=args.max_checkpoints_to_keep
    )

    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume_dir:
        resume_manager = create_checkpoint_manager(args.resume_dir, max_to_keep=1)
        try:
            start_epoch = load_checkpoint(model, optimizer, resume_manager)
            print(f"Resumed training from epoch {start_epoch}")
        except ValueError:
            print(f"No checkpoint found in {args.resume_dir}, starting from scratch")

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
        log_probs = nnx.log_softmax(logits, axis=-1)
        ece = evaluate_ece(log_probs, batch["label"])
        metrics.update(loss=loss, ece=ece, logits=logits, labels=batch["label"])
        optimizer.update(grads)

    @nnx.jit
    def eval_step(model: nnx.Module, metrics: nnx.MultiMetric, batch):
        loss, logits = loss_fn(model, batch)
        log_probs = nnx.log_softmax(logits, axis=-1)
        ece = evaluate_ece(log_probs, batch["label"])
        metrics.update(loss=loss, ece=ece, logits=logits, labels=batch["label"])

    @nnx.jit
    def update_batch_stats_step(model: nnx.Module, batch):
        _ = model(batch["image"])

    def eval_swag(swag_state: swag.SWAGState, metrics: nnx.MultiMetric, key: jax.Array):
        swag_sample_list = swag.sample_swag(args.num_swag_samples, key, swag_state)
        swa_model_list = []
        for swag_sample in swag_sample_list:
            swa_model = nnx.merge(graphdef, swag_sample, initial_batch_stats)
            if args.norm_type == "bn":
                swa_model.train()
                train_epoch_loader = train_loader.take(train_steps_per_epoch)
                for train_batch in train_epoch_loader.as_numpy_iterator():
                    update_batch_stats_step(swa_model, train_batch)
            swa_model.eval()
            swa_model_list.append(swa_model)
        for batch in val_loader.as_numpy_iterator():
            logits_list = []
            for swa_model in swa_model_list:
                _, logits = loss_fn(swa_model, batch)
                logits_list.append(logits)
            logits_list = jnp.stack(logits_list)
            logprobs = nnx.log_softmax(logits_list, axis=-1)
            ens_logprobs = nnx.logsumexp(logprobs, axis=0) - jnp.log(logprobs.shape[0])
            ens_nll = evaluate_nll(ens_logprobs, batch["label"], reduction="mean")
            ens_ece = evaluate_ece(ens_logprobs, batch["label"])
            metrics.update(
                loss=ens_nll, ece=ens_ece, logits=ens_logprobs, labels=batch["label"]
            )

    with wandb.init(project=f"resnet-swag-{args.ds_name}", config=args) as run:
        for epoch in tqdm(range(start_epoch, args.optim_num_epochs)):
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

            step = epoch + 1
            run.log(
                {
                    **train_metrics_dict,
                    **test_metrics_dict,
                    "lr": scheduler(int(optimizer.step.value)).item(),
                    "steps": optimizer.step.value,
                },
                step=step,
            )
            if epoch >= args.start_swa_epoch and (
                step % args.eval_swag_freq == 0 or step == args.optim_num_epochs
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
                run.log({**swag_test_metrics_dict}, step=step)

            # Save checkpoint with specified frequency
            if (
                step % args.checkpoint_every_n_epochs == 0
                or step == args.optim_num_epochs
            ):
                save_checkpoint(
                    checkpoint_manager,
                    model,
                    optimizer,
                    step,
                    metrics={**train_metrics_dict, **test_metrics_dict},
                )

        swag_state = optimizer.opt_state[-1]
        save_swag_state(swag_state, args.swag_save_dir, args.model_id)
        print(f"Training completed. SWAG state saved to {args.swag_save_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_depth", default=32, type=int, choices=[20, 32, 44, 56, 110]
    )
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--norm_type", default="frn", type=str, choices=["bn", "frn"])
    parser.add_argument(
        "--ds_name", default="cifar10", type=str, choices=["cifar10", "cifar100"]
    )
    parser.add_argument("--num_classes", default=10, type=int)
    parser.add_argument("--seed", default=5, type=int)

    parser.add_argument("--optim_lr", default=0.1, type=float)
    parser.add_argument("--optim_swa_lr", default=0.01, type=float)
    parser.add_argument("--optim_momentum", default=0.9, type=float)
    parser.add_argument("--optim_weight_decay", default=1e-4, type=float)
    parser.add_argument("--optim_num_epochs", default=1000, type=int)
    parser.add_argument("--warmup_epochs", default=10, type=int)
    parser.add_argument("--start_swa_epoch", default=800, type=int)
    parser.add_argument("--swag_freq", default=1, type=int)
    parser.add_argument("--swag_rank", default=20, type=int)
    parser.add_argument("--eval_swag_freq", default=10, type=int)
    parser.add_argument("--num_swag_samples", default=20, type=int)

    parser.add_argument("--save_dir", default="./checkpoint/swag", type=str)
    parser.add_argument(
        "--swag_save_dir",
        default="./checkpoint/swag_state",
        type=str,
        help="Directory to save SWAG state separately",
    )
    parser.add_argument(
        "--model_id",
        default=None,
        type=str,
        help="Model identifier for multi-model training",
    )
    parser.add_argument(
        "--max_checkpoints_to_keep",
        default=5,
        type=int,
        help="Maximum number of checkpoints to keep (older ones will be deleted)",
    )
    parser.add_argument(
        "--checkpoint_every_n_epochs",
        default=10,
        type=int,
        help="Save checkpoint every N epochs",
    )
    parser.add_argument(
        "--resume_dir",
        default=None,
        type=str,
        help="Path to checkpoint directory to resume from",
    )

    args = parser.parse_args()
    KST = timezone(timedelta(hours=9))
    now = datetime.now(KST).strftime("%Y-%m-%d_%H-%M-%S")
    args.save_dir = os.path.join(args.save_dir, now)
    args.save_dir = os.path.abspath(args.save_dir)

    if not args.model_id:
        args.swag_save_dir = os.path.join(args.swag_save_dir, now)
    args.swag_save_dir = os.path.abspath(args.swag_save_dir)

    if args.resume_dir:
        args.resume_dir = os.path.abspath(args.resume_dir)
    launch(args)


if __name__ == "__main__":
    main()
