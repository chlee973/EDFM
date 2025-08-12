from typing import Tuple
from functools import partial
import chex
import jax
import jax.numpy as jnp
import optax

from .state import SWAGDiagState, SWAGState


def sample_swag_diag(
    num_samples: int, key: chex.PRNGKey, state: SWAGDiagState, eps: float = 1e-30
) -> optax.Params:
    mean, tree_unflatten_fn = jax.flatten_util.ravel_pytree(state.mean)
    p2, _ = jax.flatten_util.ravel_pytree(state.params2)
    std = jnp.sqrt(jnp.clip(p2 - jnp.square(mean), a_min=eps))

    z = jax.random.normal(key, (num_samples, *mean.shape))

    construct_swagdiag_params = lambda z: mean + std * z
    swagdiag_params = jax.vmap(construct_swagdiag_params)(z)

    swagdiag_param_list = []
    for i in range(len(swagdiag_params)):
        swagdiag_param_list.append(tree_unflatten_fn(swagdiag_params[i]))

    return swagdiag_param_list


def sample_swag(
    num_samples: int,
    key: chex.PRNGKey,
    state: SWAGState,
    scale: float = 1.0,
    eps: float = 1e-30,
):
    mean, tree_unflatten_fn = jax.flatten_util.ravel_pytree(state.mean)
    p2, _ = jax.flatten_util.ravel_pytree(state.params2)

    std = jnp.sqrt(jnp.clip(p2 - jnp.square(mean), a_min=eps))

    dparams = jax.vmap(lambda tree: jax.flatten_util.ravel_pytree(tree)[0])(
        state.dparams
    )
    rank = dparams.shape[0]

    z1_key, z2_key = jax.random.split(key, 2)
    z1 = jax.random.normal(z1_key, (num_samples, *mean.shape))
    z2 = jax.random.normal(z2_key, (num_samples, rank))

    z1_scale = scale / jnp.sqrt(2)
    z2_scale = scale / jnp.sqrt(2 * (rank - 1))

    construct_swag_params = (
        lambda z1, z2: mean + z1_scale * std * z1 + z2_scale * jnp.matmul(dparams.T, z2)
    )
    swag_params = jax.vmap(construct_swag_params, in_axes=(0, 0))(z1, z2)

    swag_param_list = []
    for i in range(len(swag_params)):
        swag_param_list.append(tree_unflatten_fn(swag_params[i]))

    return swag_param_list
