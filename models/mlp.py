# MLP based model architecture using Flax NNX
from typing import Any
import math
from einops import rearrange
import jax.numpy as jnp
import flax.nnx as nnx


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = jnp.exp(
        -math.log(max_period) * jnp.arange(0, half, dtype=jnp.float32) / half
    )
    args = timesteps[..., None] * freqs[None]
    embedding = jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)
    if len(timesteps.shape) == 2:
        embedding = rearrange(embedding, "b n d -> b (n d)")
    if dim % 2:
        embedding = jnp.concatenate(
            [embedding, jnp.zeros_like(embedding[:, :1])], axis=-1
        )
    return embedding


class MLP(nnx.Module):
    def __init__(
        self,
        cond_feature_dim: int,
        hidden_dim: int,
        time_embed_dim: int,
        num_blocks: int,
        num_classes: int,
        droprate: float,
        time_scale: float,
        dtype: Any = jnp.float32,
        s_data: float = 1.0,
        s_noise: float = 4.0,
        *,
        rngs: nnx.Rngs,
    ):
        self.cond_feature_dim = cond_feature_dim
        self.hidden_dim = hidden_dim
        self.time_embed_dim = time_embed_dim
        self.num_blocks = num_blocks
        self.num_classes = num_classes
        self.droprate = droprate
        self.time_scale = time_scale
        self.dtype = dtype
        self.s_data = s_data
        self.s_noise = s_noise
        self.feature_dim = cond_feature_dim + num_classes
        # Initialize time embedding dense layers for each block
        self.time_dense_layers = []
        for i in range(num_blocks):
            out_features = 3 * self.feature_dim if i == 0 else 3 * hidden_dim
            self.time_dense_layers.append(
                nnx.Linear(
                    in_features=time_embed_dim,
                    out_features=out_features,
                    kernel_init=nnx.initializers.constant(0.0),
                    rngs=rngs,
                )
            )

        # Initialize MLP dense layers for each block
        self.mlp_dense_layers = []
        for i in range(num_blocks):
            in_features = self.feature_dim if i == 0 else hidden_dim
            self.mlp_dense_layers.append(
                nnx.Linear(
                    in_features=in_features,
                    out_features=hidden_dim,
                    dtype=dtype,
                    kernel_init=nnx.initializers.xavier_uniform(),
                    bias_init=nnx.initializers.normal(stddev=1e-6),
                    rngs=rngs,
                )
            )

        # Output layer
        self.output_dense = nnx.Linear(
            in_features=hidden_dim,
            out_features=num_classes,
            dtype=dtype,
            kernel_init=nnx.initializers.xavier_uniform(),
            bias_init=nnx.initializers.normal(stddev=1e-6),
            rngs=rngs,
        )

        # LayerNorm layers for each block
        self.layer_norms = []
        for i in range(num_blocks):
            self.layer_norms.append(
                nnx.LayerNorm(
                    num_features=self.feature_dim,
                    use_bias=False,
                    use_scale=False,
                    rngs=rngs,
                )
            )

        # Dropout layer
        self.dropout = nnx.Dropout(rate=droprate, rngs=rngs)

    def __call__(self, x, z, t):
        x_copy = x

        c_in = 1 / jnp.sqrt(t**2 * self.s_data**2 + (1 - t) ** 2 * self.s_noise**2)
        c_skip = (t * self.s_data**2 - (1 - t) * self.s_noise**2) * c_in**2
        c_out = self.s_data * self.s_noise * c_in

        x *= c_in[..., None]
        t = jnp.log(self.time_scale * (1 - t) + 1e-12) / 4

        x = jnp.concatenate([x, z], axis=-1)

        t_skip = timestep_embedding(t, self.time_embed_dim)

        # MLP Residual blocks
        for i in range(self.num_blocks):
            x_skip = x

            # Time embedding processing
            t_processed = self.time_dense_layers[i](t_skip)
            t_processed = nnx.silu(t_processed)
            shift_mlp, scale_mlp, gate_mlp = jnp.split(t_processed, 3, axis=-1)

            # Layer normalization and modulation
            x = self.layer_norms[i](x)
            x = x * (1 + scale_mlp) + shift_mlp

            # MLP processing
            x = self.mlp_dense_layers[i](x)
            x = nnx.gelu(x)

            # Residual connection with gating
            x = x_skip + (gate_mlp * x) if i > 0 else x

            # Dropout
            x = self.dropout(x)

        # Output layer
        x = self.output_dense(x)

        return c_skip[..., None] * x_copy + c_out[..., None] * x
