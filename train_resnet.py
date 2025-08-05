import os
import argparse
from datetime import datetime

import jax
import optax
from flax import nnx
import wandb
from tqdm import tqdm
import orbax.checkpoint as orbax
from typing import Type
import resnet
from dataloader_tfds import build_dataloader

def save_checkpoint(model: nnx.Module, path: str):
  state = nnx.state(model)
  # Save the parameters
  checkpointer = orbax.PyTreeCheckpointer()
  checkpointer.save(f'{path}/state', state)
  
def load_checkpoint(model_arch: Type[nnx.Module], path: str) -> nnx.Module:
  # create that model with abstract shapes
  model = nnx.eval_shape(lambda: model_arch(rngs=nnx.Rngs(0)))
  state = nnx.state(model)
  # Load the parameters
  checkpointer = orbax.PyTreeCheckpointer()
  state = checkpointer.restore(f'{path}/state', item=state)
  # update the model with the loaded state
  nnx.update(model, state)
  return model

def launch(args):
  train_loader, val_loader = build_dataloader(args.batch_size)
  
  model_arch = resnet.__dict__[f"resnet{args.model_depth}"]
  model = model_arch(rngs=nnx.Rngs(args.seed))

  train_steps_per_epoch = 50000 // args.batch_size
  schedule = optax.piecewise_constant_schedule(
      init_value = args.optim_lr,
      boundaries_and_scales={100*train_steps_per_epoch: 0.1, 150*train_steps_per_epoch: 0.1}
  )
  tx = optax.chain(
    optax.add_decayed_weights(weight_decay=args.optim_weight_decay),
    optax.sgd(schedule, args.optim_momentum),
  )

  optimizer = nnx.Optimizer(
    model, tx, wrt=nnx.Param
  )
  metrics = nnx.MultiMetric(
    accuracy=nnx.metrics.Accuracy(),
    loss=nnx.metrics.Average('loss'),
  )

  def loss_fn(model: nnx.Module, batch):
    logits = model(batch['image'])
    loss = optax.softmax_cross_entropy_with_integer_labels(
      logits=logits, labels=batch['label']
    ).mean()
    return loss, logits

  @nnx.jit
  def train_step(model: nnx.Module, optimizer: nnx.Optimizer, metrics: nnx.MultiMetric, batch):
    """Train for a single step."""
    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(model, batch)
    metrics.update(loss=loss, logits=logits, labels=batch['label'])
    optimizer.update(grads)

  @nnx.jit
  def eval_step(model: nnx.Module, metrics: nnx.MultiMetric, batch):
    loss, logits = loss_fn(model, batch)
    metrics.update(loss=loss, logits=logits, labels=batch['label'])

  with wandb.init(project="resnet-cifar10",config=args) as run:
    train_iter = iter(train_loader.as_numpy_iterator())
    for epoch in tqdm(range(args.optim_num_epochs)):
      model.train()
      for step in range(train_steps_per_epoch):
        batch = next(train_iter)
        train_step(model, optimizer, metrics, batch)
      # Log the train metrics.
      train_metrics_dict = {f"train/{metric}": value for metric, value in metrics.compute().items()}
      metrics.reset()  # Reset the metrics for the test set.

      # Compute the metrics on the test set after each training epoch.
      model.eval()
      for batch in val_loader.as_numpy_iterator():
        eval_step(model, metrics, batch)
      # Log the test metrics.
      test_metrics_dict = {f"test/{metric}": value for metric, value in metrics.compute().items()}
      metrics.reset()  # Reset the metrics for the next training epoch.
      run.log({**train_metrics_dict, **test_metrics_dict, "lr": schedule(int(optimizer.step.value)).item(), "steps": optimizer.step.value})
      save_checkpoint(model, os.path.abspath(f"{args.save_dir}/{epoch+1:03d}"))
  
def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--model_depth', default=20, type=int,
                        choices=[20, 32, 44, 56, 110])
  parser.add_argument('--batch_size', default=128, type=int)
  parser.add_argument('--seed', default=42, type=int)
  parser.add_argument("--optim_lr", default=0.1, type=float)
  parser.add_argument("--optim_momentum", default=0.9, type=float)
  parser.add_argument("--optim_weight_decay", default=1e-4, type=float)
  parser.add_argument("--optim_num_epochs", default=200, type=int)
  parser.add_argument('--save_dir', default='./checkpoint/', type=str)
  
  
  args = parser.parse_args()
  now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  save_dir = os.path.join(args.save_dir, now)
  args.save_dir = save_dir
  launch(args)

if __name__ == "__main__":
  main()
  