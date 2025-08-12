import jax
import jax.numpy as jnp


def evaluate_nll(confidences, true_labels, log_input=True, eps=1e-8, reduction="mean"):
    """
    Args:
        confidences (Array): An array with shape [N, K,].
        true_labels (Array): An array with shape [N,].
        log_input (bool): Specifies whether confidences are already given as log values.
        eps (float): Small value to avoid evaluation of log(0) when log_input is False.
        reduction (str): Specifies the reduction to apply to the output.

    Returns:
        An array of negative log-likelihood with shape [1,] when reduction in ["mean", "sum",], or
        raw negative log-likelihood values with shape [N,] when reduction in ["none",].
    """
    log_confidences = confidences if log_input else jnp.log(confidences + eps)
    true_target = jax.nn.one_hot(true_labels, num_classes=log_confidences.shape[1])
    raw_results = -jnp.sum(true_target * log_confidences, axis=-1)
    if reduction == "none":
        return raw_results
    elif reduction == "mean":
        return jnp.mean(raw_results)
    elif reduction == "sum":
        return jnp.sum(raw_results)
    else:
        raise NotImplementedError(f'Unknown reduction="{reduction}"')


def evaluate_acc(confidences, true_labels, log_input=True, eps=1e-8, reduction="mean"):
    """
    Args:
        confidences (Array): An array with shape [N, K,].
        true_labels (Array): An array with shape [N,].
        log_input (bool, unused): Specifies whether confidences are already given as log values.
        eps (float, unused): Small value to avoid evaluation of log(0) when log_input is False.
        reduction (str): Specifies the reduction to apply to the output.

    Returns:
        An array of accuracy with shape [1,] when reduction in ["mean", "sum",], or raw accuracy
        values with shape [N,] when reduction in ["none",].
    """
    pred_labels = jnp.argmax(confidences, axis=1)
    raw_results = jnp.equal(pred_labels, true_labels)
    if reduction == "none":
        return raw_results
    elif reduction == "mean":
        return jnp.mean(raw_results)
    elif reduction == "sum":
        return jnp.sum(raw_results)
    else:
        raise NotImplementedError(f'Unknown reduction="{reduction}"')


def evaluate_ece(
    log_probs: jnp.ndarray, labels: jnp.ndarray, num_bins: int = 15
) -> jnp.ndarray:
    probs = jnp.exp(log_probs)
    confidences = jnp.max(probs, axis=1)  # (B,)
    predictions = jnp.argmax(probs, axis=1)  # (B,)
    accuracies = predictions == labels  # (B,)
    n = log_probs.shape[0]

    bin_boundaries = jnp.linspace(0.0, 1.0, num_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    def bin_ece(bin_lower, bin_upper):
        in_bin = (confidences >= bin_lower) & (confidences < bin_upper)  # (B,)
        prop_in_bin = jnp.mean(in_bin.astype(jnp.float32))

        # Avoid division by zero
        avg_conf = jnp.sum(jnp.where(in_bin, confidences, 0.0)) / jnp.maximum(
            jnp.sum(in_bin), 1.0
        )
        avg_acc = jnp.sum(
            jnp.where(in_bin, accuracies.astype(jnp.float32), 0.0)
        ) / jnp.maximum(jnp.sum(in_bin), 1.0)

        return prop_in_bin * jnp.abs(avg_conf - avg_acc)

    ece = jnp.sum(jax.vmap(bin_ece)(bin_lowers, bin_uppers))
    return ece
