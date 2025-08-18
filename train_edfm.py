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
from typing import Type, Optional
from models.resnet import resnet32
from models.mlp import MLP
from models.flowmatching import FlowMatching
from dataloader_tfds import build_dataloader
import swag
from eval import evaluate_nll, evaluate_ece


def create_checkpoint_manager(
    save_dir: str, max_to_keep: int = 5
) -> ocp.CheckpointManager:
    """Create a checkpoint manager with automatic cleanup."""
    options = ocp.CheckpointManagerOptions(
        max_to_keep=max_to_keep,
        create=True,
    )
    return ocp.CheckpointManager(save_dir, ocp.PyTreeCheckpointer(), options)


def save_checkpoint(
    checkpoint_manager: ocp.CheckpointManager,
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
    checkpoint_manager: ocp.CheckpointManager,
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


def load_resnet(
    abstract_resnet: nnx.Module, checkpoint_manager: ocp.CheckpointManager
) -> int:
    """Load checkpoint into existing model and optimizer objects and return step."""
    # Get the latest step if not specified
    step = checkpoint_manager.latest_step()
    if step is None:
        raise ValueError("No checkpoint found")

    # Load the checkpoint
    restored_state = checkpoint_manager.restore(step)
    # Update model with loaded state
    resnet_graphdef, abstract_state = nnx.split(abstract_resnet)

    # Remove intermediate values (like 'feature' from nnx.sow) from restored model state
    model_state = restored_state["model"]
    model_state = {k: v for k, v in model_state.items() if k != "feature"}
    # Filter out nnx.Intermediate values that cause leaf node mismatch
    resnet = nnx.merge(resnet_graphdef, model_state)
    return resnet


def save_swag_state(
    swag_state: swag.SWAGState,
    save_path: str,
    model_id: Optional[str] = None,
):
    """Save SWAG state separately for inference."""
    if model_id:
        save_path = os.path.join(save_path, f"model_{model_id}")

    checkpoint_manager = ocp.CheckpointManager(
        save_path,
        ocp.PyTreeCheckpointer(),
        ocp.CheckpointManagerOptions(max_to_keep=1, create=True),
    )
    checkpoint_manager.save(0, {"swag_state": swag_state})
    print(f"SWAG state saved to {save_path}")


def load_swag_state(
    load_path: str,
) -> swag.SWAGState:  # Change return type annotation
    """Load SWAG state for inference."""
    checkpoint_manager = ocp.CheckpointManager(
        load_path,
        ocp.PyTreeCheckpointer(),
        ocp.CheckpointManagerOptions(max_to_keep=1, create=False),
    )
    restored_state = checkpoint_manager.restore(0)
    # Convert to proper SWAGState object
    swag_state_dict = restored_state["swag_state"]
    return swag.SWAGState(**swag_state_dict)


def launch(args):
    fm_key, score_key, train_key, eval_key = jax.random.split(
        jax.random.key(args.seed), 4
    )
    # Dataloader
    print("Building Dataloader..")
    train_loader, val_loader = build_dataloader(args.batch_size)
    train_steps_per_epoch = 50000 // args.batch_size

    # Prepare teachers
    print("Preparing teachers..")
    swag_state_list = []
    for swag_state_dir in os.listdir(args.teacher_dir):
        swag_state = load_swag_state(f"{args.teacher_dir}/{swag_state_dir}")
        swag_state_list.append(swag_state)

    # Define and load resnet (feature extractor)
    print("Loading resnet..")
    abstract_resnet = nnx.eval_shape(
        lambda: resnet32(norm_type=args.norm_type, rngs=nnx.Rngs(0))
    )
    resnet_graphdef, _ = nnx.split(abstract_resnet)
    load_resnet_manager = create_checkpoint_manager(args.resnet_dir, max_to_keep=1)
    resnet = load_resnet(abstract_resnet, load_resnet_manager)

    # Build score net
    print("Building score net..")
    score_net = MLP(
        cond_feature_dim=64,
        hidden_dim=256,
        time_embed_dim=32,
        num_blocks=8,
        num_classes=10,
        droprate=0.5,
        time_scale=1000.0,
        rngs=nnx.Rngs(score_key),
    )

    # Initialize model, optimizers
    model = FlowMatching(
        resnet=resnet,
        score=score_net,
        noise_var=4,
        num_classes=10,
        eps=0.001,
        base=3,
        rngs=nnx.Rngs(fm_key),
    )
    scheduler = optax.join_schedules(
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
        optax.add_decayed_weights(args.optim_weight_decay),
        optax.sgd(learning_rate=scheduler, momentum=args.optim_momentum),
    )
    score_params = nnx.All(nnx.Param, nnx.PathContains("score"))
    optimizer = nnx.Optimizer(model, tx, wrt=score_params)

    train_metrics = nnx.MultiMetric(
        loss=nnx.metrics.Average("loss"),
    )

    test_metrics = nnx.MultiMetric(
        ens_nll=nnx.metrics.Average("ens_nll"),
        ens_ece=nnx.metrics.Average("ens_ece"),
        ens_accuracy=nnx.metrics.Accuracy(),
    )

    checkpoint_manager = create_checkpoint_manager(
        args.save_dir, max_to_keep=args.max_checkpoints_to_keep
    )

    @nnx.jit
    def train_step(
        key,
        model: FlowMatching,
        optimizer: nnx.Optimizer,
        metrics: nnx.MultiMetric,
        batch,
    ):
        # Split keys for different random operations
        choice_key, swag_sample_key, loss_key = jax.random.split(key, 3)

        # Sample from SWAG ensemble
        swag_state_idx = jax.random.randint(
            choice_key, shape=(), minval=0, maxval=len(swag_state_list)
        )
        branches = tuple((lambda s=s: s) for s in swag_state_list)
        swag_state = jax.lax.switch(swag_state_idx, branches)
        swag_sample = swag.sample_swag_diag(1, swag_sample_key, swag_state)[0]
        swa_model = nnx.merge(resnet_graphdef, swag_sample)
        swa_model.eval()

        # Get teacher logits
        l_1 = swa_model(batch["image"])

        # Define loss function with external key
        def loss_fn(model: FlowMatching):
            # Pass key to get_loss for random sampling
            loss = model.get_loss(loss_key, l_1, batch["image"])
            return loss.mean()

        # Compute gradient using nnx.value_and_grad
        diff_state = nnx.DiffState(0, score_params)
        loss, grads = nnx.value_and_grad(loss_fn, argnums=diff_state)(model)
        # Update metrics
        metrics.update(loss=loss)

        # Update optimizer
        optimizer.update(grads)

    @nnx.jit
    def mixup_step(key, batch):
        x = batch["image"]
        y = batch["label"]
        a = args.mixup_alpha
        beta_key, perm_key = jax.random.split(key)

        lamda = jnp.where(a > 0, jax.random.beta(beta_key, a, a), 1)

        perm_x = jax.random.permutation(perm_key, x)
        perm_y = jax.random.permutation(perm_key, y)
        mixed_x = (1 - lamda) * x + lamda * perm_x
        mixed_y = jnp.where(lamda < 0.5, y, perm_y)

        batch["image"] = mixed_x
        batch["label"] = mixed_y
        return batch

    @nnx.jit
    def eval_step(key, model: nnx.Module, metrics: nnx.MultiMetric, batch):
        logits = model.sample_logit(
            key, batch["image"], args.sample_num_steps, args.num_swag_samples
        )
        logprobs = nnx.log_softmax(logits, axis=-1)
        ens_logprobs = nnx.logsumexp(logprobs, axis=1) - jnp.log(logprobs.shape[1])
        ens_nll = evaluate_nll(ens_logprobs, batch["label"], reduction="mean")
        ens_ece = evaluate_ece(ens_logprobs, batch["label"])
        metrics.update(
            ens_nll=ens_nll, ens_ece=ens_ece, logits=ens_logprobs, labels=batch["label"]
        )

    with wandb.init(project="edfm-cifar10", config=args) as run:
        for epoch in tqdm(range(args.optim_num_epochs)):
            model.train()
            train_epoch_loader = train_loader.take(train_steps_per_epoch)
            for batch in train_epoch_loader.as_numpy_iterator():
                train_key, mixup_key, sub_key = jax.random.split(train_key, 3)
                if args.mixup_alpha > 0:
                    batch = mixup_step(mixup_key, batch)
                train_step(sub_key, model, optimizer, train_metrics, batch)
            # Log the train metrics.
            train_metrics_dict = {
                f"train/{metric}": value
                for metric, value in train_metrics.compute().items()
            }
            train_metrics.reset()  # Reset the metrics for the test set.

            # Compute the metrics on the test set after each training epoch.
            model.eval()
            for batch in val_loader.as_numpy_iterator():
                eval_key, sub_key = jax.random.split(eval_key)
                eval_step(sub_key, model, test_metrics, batch)
            # Log the test metrics.
            test_metrics_dict = {
                f"test/{metric}": value
                for metric, value in test_metrics.compute().items()
            }
            test_metrics.reset()  # Reset the metrics for the next training epoch.
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
    parser.add_argument("--num_swag_samples", default=32, type=int)
    parser.add_argument("--sample_num_steps", default=7, type=int)
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
