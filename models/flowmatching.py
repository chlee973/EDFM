import jax
import jax.numpy as jnp
from flax import nnx
from typing import Sequence
import functools
import optax


class FlowMatching(nnx.Module):
    def __init__(
        self,
        resnet: nnx.Module,
        score: nnx.Module,
        noise_var: float,
        num_classes: int,
        eps: float,
        base: float,
        *,
        rngs: nnx.Rngs
    ):
        self.resnet = resnet
        self.score = score
        self.noise_var = noise_var
        self.num_classes = num_classes
        self.eps = eps
        self.base = base
        self.rngs = rngs

    def get_loss(self, key, l_1, x):
        """
        Compute CFM loss using external key

        Args:
            l_1: teacher logits
            x: input images
            key: JAX random key for sampling
        """
        # Split key for different sampling operations
        t_key, noise_key = jax.random.split(key)

        # Sample t
        t = self.sample_timestep(t_key, l_1.shape[0])

        # Sample noise
        l_0 = jax.random.normal(noise_key, l_1.shape) * self.noise_var

        # Compute conditional flow / velocity
        t_expanded = t[:, None]
        l_t = t_expanded * l_1 + (1 - t_expanded) * l_0
        u_t = (l_1 - l_t) / (1 - t_expanded + 1e-8)  # Add small epsilon for stability

        # Get features from resnet
        _, feature = self.resnet(x, get_feature=True)

        # Predict velocity using score network
        src = self.score(l_t, feature, t)
        tgt = u_t

        # Compute L2 loss
        diff = src - tgt
        loss = jnp.sum(diff**2, axis=-1)
        return loss

    def sample_timestep(self, key, batch_size):
        """
        Sample timestep using external key

        Args:
            batch_size: number of samples
            key: JAX random key
        """
        u = jax.random.uniform(key, (batch_size,))
        t = jnp.log(1 + (self.base ** (1 - self.eps) - 1) * u) / jnp.log(self.base)
        return t

    def sample_logit(self, key, x, steps, num_samples):
        a = 0.7
        timesteps = jnp.array([(1 - a**i) / (1 - a**steps) for i in range(steps + 1)])
        _, feature = self.resnet(x, get_feature=True)
        feature = feature.repeat(num_samples, axis=0)

        def batch_mul(a, b):
            return jax.vmap(lambda a, b: a * b)(a, b)

        @jax.jit
        def euler_solver(n, l_n):
            current_t = jnp.array([timesteps[n]])
            current_t = jnp.tile(current_t, [l_n.shape[0]])

            next_t = jnp.array([timesteps[n + 1]])
            next_t = jnp.tile(next_t, [l_n.shape[0]])

            eps = self.score(l_n, feature, t=current_t)
            euler_l_n = l_n + batch_mul(next_t - current_t, eps)
            return euler_l_n, eps

        @jax.jit
        def heun_solver(n, l_n):
            current_t = jnp.array([timesteps[n]])
            current_t = jnp.tile(current_t, [l_n.shape[0]])

            next_t = jnp.array([timesteps[n + 1]])
            next_t = jnp.tile(next_t, [l_n.shape[0]])

            euler_l_n, eps = euler_solver(n, l_n)
            eps2 = self.score(euler_l_n, feature, t=next_t)
            heun_l_n = l_n + batch_mul((next_t - current_t) / 2, eps + eps2)
            return heun_l_n

        l_0 = (
            jax.random.normal(key, shape=(feature.shape[0], self.num_classes))
            * self.noise_var
        )
        val = l_0
        for i in range(0, steps - 1):
            val = heun_solver(i, val)
        val, _ = euler_solver(steps - 1, val)

        logits = val.reshape(-1, num_samples, self.num_classes)
        # Shape: (batch_size, num_samples, num_classes)
        return logits
