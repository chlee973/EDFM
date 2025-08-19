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
from typing import Optional, List
import models.resnet as resnet
from dataloader_tfds import build_dataloader
from eval import evaluate_ece
import swag
from train_edfm import load_swag_state


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


def launch(args):
    swag_sample_key = jax.random.key(args.seed)
    train_loader, val_loader, train_steps_per_epoch = build_dataloader(
        args.ds_name, args.batch_size, args.seed
    )

    model_arch = resnet.__dict__[f"resnet{args.model_depth}"]
    model = model_arch(
        norm_type=args.norm_type,
        width_factor=args.model_width_factor,
        num_classes=args.num_classes,
        rngs=nnx.Rngs(args.seed),
    )
    resnet_graphdef, _ = nnx.split(model)
    print("Preparing teachers..")
    swag_state_list = []
    for swag_state_dir in os.listdir(args.teacher_dir):
        swag_state = load_swag_state(f"{args.teacher_dir}/{swag_state_dir}")
        swag_state_list.append(swag_state)

    swa_param_list = []
    for idx, swag_state in enumerate(swag_state_list):
        key = jax.random.fold_in(swag_sample_key, idx)
        swag_samples = swag.sample_swag_diag(args.num_swag_samples, key, swag_state)
        swa_param_list.extend(swag_samples)

    swa_model_list = []
    for swa_param in swa_param_list:
        swa_model = nnx.merge(resnet_graphdef, swa_param)
        swa_model.eval()
        swa_model_list.append(swa_model)

    schedule = optax.join_schedules(
        schedules=[
            optax.linear_schedule(
                init_value=0.01 * args.optim_lr,
                end_value=args.optim_lr,
                transition_steps=args.warmup_epochs * train_steps_per_epoch,
            ),
            optax.cosine_decay_schedule(
                init_value=args.optim_lr,
                decay_steps=(args.optim_num_epochs - args.warmup_epochs)
                * train_steps_per_epoch,
            ),
        ],
        boundaries=[
            args.warmup_epochs * train_steps_per_epoch,
        ],
    )
    tx = optax.chain(
        optax.add_decayed_weights(weight_decay=args.optim_weight_decay),
        optax.sgd(schedule, args.optim_momentum),
    )

    optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)
    metrics = nnx.MultiMetric(
        loss=nnx.metrics.Average("loss"),
        accuracy=nnx.metrics.Accuracy(),
        nll=nnx.metrics.Average("nll"),
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

    def train_loss_fn(model: nnx.Module, batch):
        student_logits = model(batch["image"])
        teacher_logits_list = []
        for swa_model in swa_model_list:
            teacher_logits = swa_model(batch["image"])
            teacher_logits_list.append(teacher_logits)
        teacher_logits = jnp.stack(teacher_logits_list, axis=0)
        teacher_logprobs = nnx.log_softmax(teacher_logits, axis=-1)
        teacher_ens_logprobs = nnx.logsumexp(teacher_logprobs, axis=0) - jnp.log(
            teacher_logprobs.shape[0]
        )
        student_logprobs = nnx.log_softmax(student_logits, axis=-1)
        soft_loss = optax.losses.kl_divergence_with_log_targets(
            student_logprobs, teacher_ens_logprobs, axis=-1
        ).mean()
        hard_loss = optax.softmax_cross_entropy_with_integer_labels(
            logits=student_logits, labels=batch["label"]
        ).mean()
        loss = 0.9 * soft_loss + 0.1 * hard_loss
        return loss, (student_logits, hard_loss)

    def loss_fn(model: nnx.Module, batch):
        logits = model(batch["image"])
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits=logits, labels=batch["label"]
        ).mean()
        return loss, logits

    @nnx.jit
    def train_step(
        model: nnx.Module,
        optimizer: nnx.Optimizer,
        metrics: nnx.MultiMetric,
        batch,
    ):
        """Train for a single step."""
        grad_fn = nnx.value_and_grad(train_loss_fn, has_aux=True)
        (loss, (logits, nll)), grads = grad_fn(model, batch)
        logprobs = nnx.log_softmax(logits, axis=-1)
        ece = evaluate_ece(logprobs, batch["label"])
        metrics.update(
            loss=loss, nll=nll, ece=ece, logits=logits, labels=batch["label"]
        )
        optimizer.update(grads)

    @nnx.jit
    def eval_step(model: nnx.Module, metrics: nnx.MultiMetric, batch):
        nll, logits = loss_fn(model, batch)
        logprobs = nnx.log_softmax(logits, axis=-1)
        ece = evaluate_ece(logprobs, batch["label"])
        metrics.update(loss=0, nll=nll, ece=ece, logits=logits, labels=batch["label"])

    with wandb.init(
        project=f"resnet-kd-{args.ds_name}", name=args.exp_name, config=args
    ) as run:
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
            del test_metrics_dict["test/loss"]
            metrics.reset()  # Reset the metrics for the next training epoch.

            run.log(
                {
                    **train_metrics_dict,
                    **test_metrics_dict,
                    "lr": schedule(int(optimizer.step.value)).item(),
                    "opt_steps": optimizer.step.value,
                }
            )

            # Save checkpoint with specified frequency
            step = epoch + 1
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_depth", default=32, type=int, choices=[20, 32, 44, 56, 110]
    )
    parser.add_argument("--model_width_factor", default=2, type=int)
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--norm_type", default="frn", type=str, choices=["bn", "frn"])
    parser.add_argument(
        "--ds_name", default="cifar10", type=str, choices=["cifar10", "cifar100"]
    )
    parser.add_argument("--num_classes", default=10, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--optim_lr", default=0.1, type=float)
    parser.add_argument("--optim_momentum", default=0.9, type=float)
    parser.add_argument("--optim_weight_decay", default=1e-4, type=float)
    parser.add_argument("--optim_num_epochs", default=1000, type=int)
    parser.add_argument("--warmup_epochs", default=5, type=int)
    parser.add_argument("--num_swag_samples", default=3, type=int)
    parser.add_argument("--exp_name", default=None, type=str)
    parser.add_argument("--save_dir", default="./checkpoint/kd", type=str)
    parser.add_argument(
        "--teacher_dir", default="./checkpoint/multi_swag_collection", type=str
    )
    parser.add_argument(
        "--max_checkpoints_to_keep",
        default=10,
        type=int,
        help="Maximum number of checkpoints to keep (older ones will be deleted)",
    )
    parser.add_argument(
        "--checkpoint_every_n_epochs",
        default=5,
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
    if args.resume_dir:
        args.resume_dir = os.path.abspath(args.resume_dir)
    args.teacher_dir = os.path.abspath(args.teacher_dir)
    launch(args)


if __name__ == "__main__":
    main()
