from abc import ABC, abstractmethod
from typing import Tuple
import jax
import jax.numpy as jnp
from flax import nnx


def sample_simplex(key, shape, eps=1e-4):
    """
    Uniformly sample from a simplex.
    :param sizes: sizes of the array to be returned
    :param eps: small float to avoid instability
    :param key: JAX random key for reproducible sampling
    :return: Array of shape sizes, with values summing to 1
    """
    if key is None:
        key = jax.random.PRNGKey(0)
    x = jax.random.exponential(key, shape=shape)
    p = x / jnp.sum(x, axis=-1, keepdims=True)
    p = jnp.clip(p, eps, 1 - eps)
    return p / jnp.sum(p, axis=-1, keepdims=True)


class CategoricalFlow(nnx.Module, ABC):
    """
    Base class for categorical flow models.
    The model follows the Riemannian flow matching framework to learn a vector field
    on different induced Riemannian geometries on the probability simplex.

    :param feature_extractor: feature extractor model
    :param data_dim: dimension of the data points
    :param max_t: maximum timestep for the flow
    :param ot: whether to use optimal transport during training
    :param eps: small float to avoid instability
    """

    def __init__(
        self,
        feature_extractor,
        encoder,
        n_class,
        max_t=1.0,
        eps=1e-4,
        *,
        rngs: nnx.Rngs
    ):
        self.feature_extractor = feature_extractor
        self.encoder = encoder
        self.n_class = n_class
        self.max_t = max_t
        self.eps = eps
        self._v = None  # for Hutchinson's trace estimator
        self.rngs = rngs

    #################################################
    # Helper functions                              #
    #################################################

    @staticmethod
    def sample_prior(key, shape, eps=1e-4):
        """
        Sample from the prior noise distribution on the simplex.
        :param shape: shape of the Array to be returned
        :param eps: small float to avoid instability
        :return: Array of shape size, with values summing to 1
        """
        return sample_simplex(key, shape, eps=eps)

    @classmethod
    def sample_simplex_linear(cls, p, t, eps=1e-4):
        """
        Sample from a simplex with linearly decreasing L2 distance
        :param p: one-hot probability vector, Array of shape (..., n)
        :param t: timestep between 0 and 1
        :param eps: small float to avoid instability
        :return:
            p_bar: sampled values, same size as p, with values summing to 1
            log_prob: log probability of the sampled values
        """
        samples = sample_simplex(p.shape, eps=eps)
        p_bar = samples * (1 - t) + p * t
        log_prob = cls.prior_logp0(samples, eps) - jnp.log(1 - t) * (p.shape[-1] - 1)
        return p_bar, log_prob

    @staticmethod
    def preprocess(p):
        """
        Preprocess the data point of the simplex.
        :param p: data point on the simplex
        :return: preprocessed data point of the same shape
        """
        return p

    @staticmethod
    def postprocess(p_hat):
        """
        Postprocess the data point to get back to the simplex.
        :param p_hat: processed data point
        :return: data point on the simplex of the same shape
        """
        return p_hat

    #################################################
    # Riemannian manifold operations                #
    #################################################

    @classmethod
    def proj_x(cls, x, eps=0.0):
        """
        Project the data point to the manifold.
        :param x: data point
        :param eps: small float to avoid instability
        :return: projected data point of the same shape
        """
        return x

    @classmethod
    def proj_vf(cls, vf, pt):
        """
        Project the vector field to the tangent space at a specific data point.
        :param vf: vector field
        :param pt: data point at which the tangent space is defined
        :return: projected vector field of the same shape
        """
        return vf

    @classmethod
    @abstractmethod
    def dist(cls, p, q, eps=0.0):
        """
        Geodesic distance between two data points.
        :param p: first data point, Array of shape (..., n)
        :param q: second data point, Array of shape (..., n)
        :param eps: small float to avoid instability
        :return: distance between p and q, Array of shape (...)
        """
        pass

    @classmethod
    @abstractmethod
    def norm2(cls, p, u, eps=0.0):
        """
        Squared Riemannian norm of a vector field in the tangent space.
        :param p: data point at which the tangent space is defined, Array of shape (..., n)
        :param u: vector field, Array of shape (..., n)
        :param eps: small float to avoid instability
        :return: squared Riemannian norm of u at p, Array of shape (...)
        """
        pass

    @classmethod
    @abstractmethod
    def exp(cls, p, u, eps=0.0):
        """
        Exponential map exp_p(u).
        :param p: data point at which the tangent space is defined, Array of shape (..., n)
        :param u: vector field, Array of shape (..., n)
        :param eps: small float to avoid instability
        :return: exponential map, Array of shape (..., n)
        """
        pass

    @classmethod
    @abstractmethod
    def log(cls, p, q, eps=0.0):
        """
        Logarithmic map log_p(q).
        :param p: data point at which the tangent space is defined, Array of shape (..., n)
        :param q: target data point, Array of shape (..., n)
        :param eps: small float to avoid instability
        :return: logarithmic map, Array of shape (..., n)
        """
        pass

    @classmethod
    def interpolate(cls, p, q, t, eps=0.0):
        """
        Interpolate between two data points.
        :param p: source data point, Array of shape (..., n)
        :param q: target data point, Array of shape (..., n)
        :param t: timestep between 0 and 1, Array of shape (...)
        :param eps: small float to avoid instability
        :return: interpolant at timestep t, Array of shape (..., n)
        """
        return cls.exp(p, t[:, None] * cls.log(p, q, eps), eps)

    @classmethod
    def vecfield(cls, p, q, t, eps=0.0) -> Tuple[jax.Array, jax.Array]:
        """
        Vector field at timestep t.
        :param p: source data point, Array of shape (..., n)
        :param q: target data point, Array of shape (..., n)
        :param t: timestep between 0 and 1, Array of shape (...)
        :param eps: small float to avoid instability
        :return:
            pt: interpolant at timestep t, Arrays of shape (..., n)
            vf: vector field, Arrays of shape (..., n)
        """
        pt = cls.interpolate(p, q, t, eps)
        vf = cls.log(pt, q, eps) / (1 - t)[:, None]
        return pt, vf

    #################################################
    # Forward and loss functions                    #
    #################################################

    def __call__(self, t, pt, cond_input, feature_extraction=True):
        """
        predict vector field at time t
        :param t: timestep between 0 and 1, Array of shape (B,) or a single float
        :param pt: interpolant at timestep t, Array of shape (B, n)
        :return: predicted vector field, Array of shape (B, n)
        """
        if feature_extraction:
            _, cond_input = self.feature_extractor(cond_input, get_feature=True)
        vf = self.encoder(pt, cond_input, t)
        return self.proj_vf(vf, pt)

    def get_loss(self, key, p, cond_input):
        """
        Compute the Riemannian flow matching loss.
        :param p: ground truth categorical distributions, Array of shape (B, n)
        :param cond_args: optional conditional arguments
        :return: loss
        """
        noise_key, t_key = jax.random.split(key, 2)
        noise = self.sample_prior(noise_key, p.shape)
        t = jax.random.uniform(t_key, shape=(p.shape[0],)) * self.max_t
        pt, vf = self.vecfield(self.preprocess(noise), self.preprocess(p), t, self.eps)
        pred_vf = self(t, pt, cond_input)
        loss = self.norm2(pt, pred_vf - vf, self.eps).mean()
        return loss

    #################################################
    # Sampling                                      #
    #################################################

    def sample(self, key, n_sample, n_step, cond_input):
        """
        Sampling using heun method.
        :param key: JAX PRNG key
        :param n_sample: number of samples
        :param n_step: number of Euler steps
        :param cond_args: optional conditional arguments
        :param return_traj: whether to return the whole sampling trajectory
        :return: sampled data points of shape (n_sample, D, n) or (n_step, n_sample, D, n)
        """
        _, cond_feature = self.feature_extractor(cond_input, get_feature=True)
        cond_feature = cond_feature.repeat(n_sample, axis=0)
        timesteps = jnp.linspace(0, 1, n_step + 1)

        def batch_mul(a, b):
            return jax.vmap(lambda a, b: a * b)(a, b)

        @jax.jit
        def euler_solver(n, p_n):
            current_t = jnp.array([timesteps[n]])
            current_t = jnp.tile(current_t, [p_n.shape[0]])

            next_t = jnp.array([timesteps[n + 1]])
            next_t = jnp.tile(next_t, [p_n.shape[0]])

            pred_vf = self(current_t, p_n, cond_feature, feature_extraction=False)
            euler_p_n = self.exp(p_n, batch_mul(next_t - current_t, pred_vf), self.eps)
            euler_p_n = self.proj_x(euler_p_n)
            return euler_p_n, pred_vf

        @jax.jit
        def heun_solver(n, p_n):
            current_t = jnp.array([timesteps[n]])
            current_t = jnp.tile(current_t, [p_n.shape[0]])

            next_t = jnp.array([timesteps[n + 1]])
            next_t = jnp.tile(next_t, [p_n.shape[0]])

            euler_p_n, pred_vf = euler_solver(n, p_n)
            pred_vf2 = self(next_t, euler_p_n, cond_feature, feature_extraction=False)
            heun_p_n = self.exp(
                p_n, batch_mul((next_t - current_t) / 2, pred_vf + pred_vf2), self.eps
            )
            heun_p_n = self.proj_x(heun_p_n)
            return heun_p_n

        p0 = self.preprocess(
            self.sample_prior(key, shape=(cond_feature.shape[0], self.n_class))
        )
        p = p0
        for i in range(0, n_step - 1):
            p = heun_solver(i, p)
        p, _ = euler_solver(n_step - 1, p)
        p = self.postprocess(p)
        p = p.reshape(-1, n_sample, self.n_class)
        # Shape: (batch_size, n_sample, n_class)
        return p


class SimplexCategoricalFlow(CategoricalFlow):
    r"""
    Naive implementation of the Fisher metric induced Riemannian geometry on the probability simplex.
    The metric is not defined at the boundary of the simplex.
    In order to be comparable to LinearFM, the divergence calculated here is also Euclidean.
    """

    @classmethod
    def proj_x(cls, x, eps=0.0):
        x = jnp.clip(x, eps, 1 - eps)
        return x / jnp.sum(x, axis=-1, keepdims=True)

    @classmethod
    def proj_vf(cls, vf, pt):
        return vf - jnp.mean(vf, axis=-1, keepdims=True)

    @classmethod
    def dist(cls, p, q, eps=1e-2):
        return 2 * jnp.arccos(jnp.clip(jnp.sum(jnp.sqrt(p * q), axis=-1), 0, 1 - eps))

    @classmethod
    def norm2(cls, p, u, eps=1e-2):
        mask = (p > eps).astype(jnp.float32)
        return jnp.sum(jnp.square(u) / jnp.clip(p, a_min=eps) * mask, axis=-1)

    @classmethod
    def exp(cls, p, u, eps=1e-2):
        s = jnp.sqrt(p)
        xs = u / jnp.clip(s, a_min=eps) / 2
        theta = jnp.linalg.norm(xs, axis=-1, keepdims=True)
        return (s * jnp.cos(theta) + xs * jnp.sinc(theta / jnp.pi)) ** 2

    @classmethod
    def log(cls, p, q, eps=1e-2):
        z = jnp.sqrt(p * q)
        s = jnp.sum(z, axis=-1, keepdims=True)
        dist = 2 * jnp.arccos(jnp.clip(s, 0, 1 - eps))
        u = dist / jnp.sqrt(jnp.clip(1 - s**2, a_min=eps)) * (z - s * p)
        return jnp.where(dist > eps, u, q - p)

    @classmethod
    def vecfield(cls, p, q, t, eps=1e-2):
        pt = cls.interpolate(p, q, t, eps)
        vf = cls.log(pt, q, eps) / (1 - t)[:, None]
        return pt, vf


class SphereCategoricalFlow(CategoricalFlow):
    r"""
    Implementation of the Fisher metric induced Riemannian geometry on the probability simplex
    by leverage the diffeomorphism between the probability simplex and the sphere.
    The Riemannian metric, exponential, and logarithm maps can be extended to the boundary of the simplex
    The mapping is an isometry up to a constant factor of 2, and the flow is more stable to train.

    .. math ::
        \pi: \Delta^{n-1} \to S^{n-1}, \quad p \mapsto \sqrt{p}
    """

    @staticmethod
    def preprocess(p):
        return jnp.sqrt(p)

    @staticmethod
    def postprocess(p_hat):
        return jnp.square(p_hat)

    @classmethod
    def proj_x(cls, x, eps=0.0):
        x = jnp.clip(x, eps, 1 - eps)
        return x / jnp.linalg.norm(x, ord=2, axis=-1, keepdims=True)

    @classmethod
    def proj_vf(cls, vf, pt):
        return vf - jnp.sum((pt * vf), axis=-1, keepdims=True) * pt

    @classmethod
    def dist(cls, p, q, eps=1e-4):
        result = jnp.sum(p * q, axis=-1)
        result = jnp.clip(result, 0, 1 - eps)
        result = jnp.acos(result)
        return result

    @classmethod
    def norm2(cls, p, u, eps=1e-4):
        return jnp.sum(jnp.square(u), axis=-1)

    @classmethod
    def exp(cls, p, u, eps=1e-4):
        u_norm = jnp.linalg.norm(u, ord=2, axis=-1, keepdims=True)
        exp = jnp.cos(u_norm) * p + jnp.sinc(u_norm / jnp.pi) * u
        return exp

    @classmethod
    def log(cls, p, q, eps=1e-4):
        x, y = p, q
        u = y - x
        u = u - jnp.sum((x * u), axis=-1, keepdims=True) * x
        u_norm = jnp.linalg.norm(u, ord=2, axis=-1, keepdims=True)
        dist = cls.dist(x, y, eps)[:, None]
        return jnp.where(dist > eps, u * dist / u_norm, u)

    @classmethod
    def vecfield(cls, p, q, t, eps=1e-4):
        x, y = p, q
        dist = cls.dist(x, y, eps)[:, None]
        xt = cls.interpolate(x, y, t, eps)
        ux = x - xt
        ux = ux - jnp.sum((xt * ux), axis=-1, keepdims=True) * xt
        ux_norm = jnp.linalg.norm(ux, ord=2, axis=-1, keepdims=True)
        uy = y - xt
        uy = uy - jnp.sum((xt * uy), axis=-1, keepdims=True) * xt
        uy_norm = jnp.linalg.norm(uy, ord=2, axis=-1, keepdims=True)
        vf = dist * jnp.where(ux_norm > eps, -ux / ux_norm, uy / uy_norm)
        return xt, vf


class LogitFlow(CategoricalFlow):
    @staticmethod
    def sample_prior(key, shape, eps=1e-4):
        l = jax.random.normal(key, shape)
        return l

    @classmethod
    def dist(cls, p, q, eps=1e-2):
        return jnp.linalg.norm(p - q, ord=2, axis=-1)

    @classmethod
    def norm2(cls, p, u, eps=1e-2):
        return jnp.sum(u**2, axis=-1)

    @classmethod
    def exp(cls, p, u, eps=1e-2):
        return p + u

    @classmethod
    def log(cls, p, q, eps=1e-2):
        return q - p

    @classmethod
    def vecfield(cls, p, q, t, eps=1e-2):
        pt = cls.interpolate(p, q, t, eps)
        vf = cls.log(pt, q, eps) / (1 - t)[:, None]
        return pt, vf
