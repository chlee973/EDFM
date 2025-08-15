import math
from typing import Any, Callable, Optional, Tuple

import jax
import jax.numpy as jnp
from flax import nnx

PRNGKey = Any
Shape = Tuple[int, ...]
Dtype = Any
Array = Any


class FilterResponseNorm(nnx.Module):
    """Filter Response Normalization (FRN; https://arxiv.org/abs/1911.09737).

    FRN normalizes the activations of the layer for each given example in a
    batch independently, rather than across a batch like Batch Normalization.

    Attributes:
        features: number of feature channels.
        epsilon: a small float to avoid dividing by zero (default: 1e-06).
        dtype: the dtype of the result (default: infer from input and params).
        param_dtype: the dtype of parameter initializers (default: float32).
        use_bias: whether to add a bias (default: True).
        use_scale: whether to multiply by a scale (default: True).
        bias_init: initializer for bias (default: zeros).
        scale_init: initializer for scale (default: ones).
        threshold_init: initializer for threshold (default: zeros).
    """

    def __init__(
        self,
        features: int,
        *,
        epsilon: float = 1e-6,
        dtype: Optional[Dtype] = None,
        param_dtype: Dtype = jnp.float32,
        use_bias: bool = True,
        use_scale: bool = True,
        bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = nnx.initializers.zeros,
        scale_init: Callable[[PRNGKey, Shape, Dtype], Array] = nnx.initializers.ones,
        threshold_init: Callable[
            [PRNGKey, Shape, Dtype], Array
        ] = nnx.initializers.zeros,
        rngs: nnx.Rngs,
    ):
        self.features = features
        self.epsilon = epsilon
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.use_bias = use_bias
        self.use_scale = use_scale

        if self.use_scale:
            self.scale = nnx.Param(scale_init(rngs.params(), (features,), param_dtype))
        else:
            self.scale = None

        if self.use_bias:
            self.bias = nnx.Param(bias_init(rngs.params(), (features,), param_dtype))
        else:
            self.bias = None

        self.threshold = nnx.Param(
            threshold_init(rngs.params(), (features,), param_dtype)
        )

    def __call__(self, inputs):
        y = inputs
        nu2 = jnp.mean(jnp.square(inputs), axis=(1, 2), keepdims=True)
        mul = jax.lax.rsqrt(nu2 + self.epsilon)

        if self.use_scale and self.scale is not None:
            scale = self.scale.value.reshape((1, 1, 1, -1))
            mul *= scale
        y *= mul

        if self.use_bias and self.bias is not None:
            bias = self.bias.value.reshape((1, 1, 1, -1))
            y += bias

        tau = self.threshold.value.reshape((1, 1, 1, -1))
        z = jnp.maximum(y, tau)

        if self.dtype is not None:
            return jnp.asarray(z, self.dtype)
        return z
