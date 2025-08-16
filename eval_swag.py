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
import models.resnet as resnet
from dataloader_tfds import build_dataloader
import swag
from eval import evaluate_ece, evaluate_nll


def save_model_state(model: nnx.Module, path: str):
    state = nnx.state(model).to_pure_dict()
    # Save the parameters
    checkpointer = orbax.PyTreeCheckpointer()
    checkpointer.save(f"{path}/model", state)


def save_opt_state(optimizer: nnx.Optimizer, path: str):
    state = nnx.state(optimizer).to_pure_dict()
    checkpointer = orbax.PyTreeCheckpointer()
    checkpointer.save(f"{path}/opt", state)


def load_model_state(model_cls, path):
    checkpointer = orbax.PyTreeCheckpointer()
    restored_pure_dict = checkpointer.restore(f"{path}/model")
    abstract_model = nnx.eval_shape(lambda: model_cls(rngs=nnx.Rngs(0)))
    graphdef, abstract_state = nnx.split(abstract_model)
    nnx.replace_by_pure_dict(abstract_state, restored_pure_dict)
    model = nnx.merge(graphdef, abstract_state)
    return model


def load_opt_state(optimizer, path):
    checkpointer = orbax.PyTreeCheckpointer()
    restored_pure_dict = checkpointer.restore(f"{path}/opt")
    graphdef, abstract_state = nnx.split(optimizer)
    nnx.replace_by_pure_dict(abstract_state, restored_pure_dict)
    optimizer = nnx.merge(graphdef, abstract_state)
    return optimizer


def launch(args):
    model_key, swag_sample_key = jax.random.split(jax.random.key(args.seed))
    train_loader, val_loader = build_dataloader(args.batch_size)

    # The saved checkpoint appears to use BatchNorm based on the structure
    # Let's force use of BatchNorm to match the checkpoint
    model_arch = resnet.__dict__[f"resnet{args.model_depth}"]
    abstract_model = nnx.eval_shape(
        lambda: model_arch(norm_type=args.norm_type, rngs=nnx.Rngs(0))
    )
    graphdef, _, initial_batch_stats = nnx.split(
        abstract_model, nnx.Param, nnx.BatchStat
    )
    swag_dir = "/home/chlee973/EDFM/checkpoint/multi_swag_collection/2025-08-16_03-26-03/swag_state_01_seed_42/2025-08-16_12-26-09/model_model_01_seed_42"
    checkpoint_manager = orbax.CheckpointManager(
        str(swag_dir),
        orbax.PyTreeCheckpointer(),
        orbax.CheckpointManagerOptions(max_to_keep=1, create=False),
    )
    restored_state = checkpoint_manager.restore(0)
    swag_state_dict = restored_state["swag_state"]
    swag_state = swag.SWAGState(**swag_state_dict)
    nnx.display(swag_state)
    train_steps_per_epoch = 50000 // args.batch_size
    metrics = nnx.MultiMetric(
        accuracy=nnx.metrics.Accuracy(),
        loss=nnx.metrics.Average("loss"),
        ece=nnx.metrics.Average("ece"),
    )

    def loss_fn(model: nnx.Module, batch):
        logits = model(batch["image"])
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits=logits, labels=batch["label"]
        ).mean()
        return loss, logits

    @nnx.jit
    def eval_step(model: nnx.Module, metrics: nnx.MultiMetric, batch):
        loss, logits = loss_fn(model, batch)
        log_probs = nnx.log_softmax(logits)
        nll = evaluate_nll(log_probs, batch["label"])
        ece = evaluate_ece(log_probs, batch["label"])
        metrics.update(loss=loss, ece=ece, logits=log_probs, labels=batch["label"])

    @nnx.jit
    def update_batch_stats_step(model: nnx.Module, batch):
        _ = model(batch["image"])

    def eval_swa(swa_state, metrics: nnx.MultiMetric):
        # Create a fresh model and update its parameters
        swa_model = nnx.merge(graphdef, swa_state.mean, initial_batch_stats)
        if args.norm_type == "bn":
            swa_model.train()
            train_epoch_loader = train_loader.take(train_steps_per_epoch)
            for train_batch in train_epoch_loader.as_numpy_iterator():
                update_batch_stats_step(swa_model, train_batch)
        swa_model.eval()
        for batch in val_loader.as_numpy_iterator():
            eval_step(swa_model, metrics, batch)

    def eval_swag(swag_state, metrics: nnx.MultiMetric, key: jax.Array):
        swag_sample_list = swag.sample_swag(args.num_swag_samples, key, swag_state)
        swa_model_list = []
        for swag_sample in swag_sample_list:
            # Create a fresh model and update its parameters
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
            logits_list = jnp.stack(logits_list, axis=0)
            logprobs = nnx.log_softmax(logits_list, axis=-1)
            ens_logprobs = nnx.logsumexp(logprobs, axis=0) - jnp.log(logprobs.shape[0])
            ens_nll = evaluate_nll(ens_logprobs, batch["label"], reduction="mean")
            ens_ece = evaluate_ece(ens_logprobs, batch["label"])
            metrics.update(
                loss=ens_nll, ece=ens_ece, logits=ens_logprobs, labels=batch["label"]
            )

    with wandb.init(project="resnet-swag-eval-cifar10", config=args) as run:
        eval_swa(swag_state, metrics)
        swa_test_metrics_dict = {
            **{f"swa/{metric}": value for metric, value in metrics.compute().items()},
            "swa/iterates": swag_state.n,
        }
        metrics.reset()
        run.log({**swa_test_metrics_dict}, step=0)
        eval_swag(swag_state, metrics, swag_sample_key)
        swag_test_metrics_dict = {
            **{f"swag/{metric}": value for metric, value in metrics.compute().items()},
            "swag/iterates": swag_state.n,
            "swag/num_samples": args.num_swag_samples,
        }
        metrics.reset()
        run.log({**swag_test_metrics_dict}, step=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_depth", default=32, type=int, choices=[20, 32, 44, 56, 110]
    )
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--norm_type", default="frn", type=str, choices=["bn", "frn"])
    parser.add_argument("--seed", default=42, type=int)

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

    parser.add_argument("--save_dir", default="./1000", type=str)

    args = parser.parse_args()
    args.save_dir = os.path.abspath(args.save_dir)
    launch(args)


if __name__ == "__main__":
    main()
