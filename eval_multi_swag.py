import os
import argparse
from datetime import datetime, timezone, timedelta

import jax
import jax.numpy as jnp
import optax
from flax import nnx
import wandb
from tqdm import tqdm
import orbax.checkpoint as ocp
from typing import Type, Optional, List
from models.resnet import resnet32
from models.mlp import MLP
from models.flowmatching import FlowMatching
from dataloader_tfds import build_dataloader
import swag
from eval import evaluate_nll, evaluate_ece
from train_edfm import load_swag_state


def launch(args):
    swag_sample_key, _, _ = jax.random.split(jax.random.key(args.seed), 3)
    # Dataloader
    print("Building Dataloader..")
    train_loader, val_loader = build_dataloader(args.batch_size)
    train_steps_per_epoch = 50000 // args.batch_size

    abstract_resnet = nnx.eval_shape(
        lambda: resnet32(norm_type=args.norm_type, rngs=nnx.Rngs(0))
    )
    resnet_graphdef, _ = nnx.split(abstract_resnet)

    # Prepare teachers
    print("Preparing teachers..")
    swag_state_list = []
    for swag_state_dir in os.listdir(args.teacher_dir):
        swag_state = load_swag_state(f"{args.teacher_dir}/{swag_state_dir}")
        swag_state_list.append(swag_state)

    swa_param_list = []
    for idx, swag_state in enumerate(swag_state_list):
        key = jax.random.fold_in(swag_sample_key, idx)
        swag_samples = swag.sample_swag_diag(20, key, swag_state)
        swa_param_list.extend(swag_samples)

    swa_model_list = []
    for swa_param in swa_param_list:
        swa_model = nnx.merge(resnet_graphdef, swa_param)
        swa_model.eval()
        swa_model_list.append(swa_model)

    metrics = nnx.MultiMetric(
        # nll=nnx.metrics.Average("nll"),
        ens_nll=nnx.metrics.Average("ens_nll"),
        ens_ece=nnx.metrics.Average("ens_ece"),
        ens_accuracy=nnx.metrics.Accuracy(),
    )

    @nnx.jit
    def loss_fn(model: nnx.Module, batch):
        logits = model(batch["image"])
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits=logits, labels=batch["label"]
        ).mean()
        return loss, logits

    def eval_step(swa_model_list: List[nnx.Module], metrics: nnx.MultiMetric, batch):
        logits_list = []
        for swa_model in swa_model_list:
            _, logits = loss_fn(swa_model, batch)
            logits_list.append(logits)
        logits = jnp.stack(logits_list, axis=1)
        logprobs = nnx.log_softmax(logits, axis=-1)
        ens_logprobs = nnx.logsumexp(logprobs, axis=1) - jnp.log(logprobs.shape[1])
        ens_nll = evaluate_nll(ens_logprobs, batch["label"], reduction="mean")
        ens_ece = evaluate_ece(ens_logprobs, batch["label"])
        metrics.update(
            ens_nll=ens_nll, ens_ece=ens_ece, logits=ens_logprobs, labels=batch["label"]
        )

    # Compute the metrics on the test set after each training epoch
    for batch in val_loader.as_numpy_iterator():
        eval_step(swa_model_list, metrics, batch)
    test_metrics_dict = {
        **{f"{metric}": value for metric, value in metrics.compute().items()},
    }
    print(test_metrics_dict)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_depth", default=32, type=int, choices=[20, 32, 44, 56, 110]
    )
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--norm_type", default="frn", type=str, choices=["bn", "frn"])
    parser.add_argument("--seed", default=4, type=int)

    parser.add_argument("--optim_lr", default=1e-4, type=float)
    parser.add_argument("--optim_momentum", default=0.9, type=float)
    parser.add_argument("--optim_weight_decay", default=5e-4, type=float)
    parser.add_argument("--optim_num_epochs", default=200, type=int)
    parser.add_argument("--warmup_epochs", default=10, type=int)
    parser.add_argument("--swag_freq", default=1, type=int)
    parser.add_argument("--swag_rank", default=20, type=int)
    parser.add_argument("--num_swag_samples", default=1, type=int)
    parser.add_argument("--mixup_alpha", default=0.4, type=float)
    parser.add_argument("--save_dir", default="./checkpoint/edfm", type=str)
    parser.add_argument("--resnet_dir", default="./checkpoint/resnet_feature", type=str)
    parser.add_argument(
        "--teacher_dir", default="./checkpoint/multi_swag_collection", type=str
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
    save_dir = os.path.join(args.save_dir, now)
    args.save_dir = os.path.abspath(save_dir)
    args.resnet_dir = os.path.abspath(args.resnet_dir)
    args.teacher_dir = os.path.abspath(args.teacher_dir)

    if args.resume_dir:
        args.resume_dir = os.path.abspath(args.resume_dir)
    launch(args)


if __name__ == "__main__":
    main()
