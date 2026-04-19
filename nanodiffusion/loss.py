"""Loss primitives shared by pretraining and SFT paths.

Paradigm-specific loss functions and training loops live under
``nanodiffusion.pretrain`` and ``nanodiffusion.sft``; this module keeps
the small per-token helpers and the time-sampling Protocol that both
paradigms depend on so neither has to import the other.
"""

from typing import Protocol, runtime_checkable

import jax
import jax.numpy as jnp
from jaxtyping import Float

from nanodiffusion.types import Logits, PRNGKeyArray, Tokens


@runtime_checkable
class TimeSampler(Protocol):
    def __call__(self, batch_size: int, *, key: PRNGKeyArray) -> jax.Array: ...


def low_discrepancy_sampler(batch_size: int, *, key: PRNGKeyArray) -> jax.Array:
    """Stratified time sampling with a single random offset. See MDLM Sec. 3.3."""
    sampling_eps = 1e-5
    u = jax.random.uniform(key, (1,))
    t_batch = (u / batch_size + jnp.arange(batch_size) / batch_size) % 1
    return (1 - sampling_eps) * t_batch + sampling_eps


def token_nll(logits: Logits, x0: Tokens) -> Float[jax.Array, " seq"]:
    """Per-position negative log-likelihood of the true tokens.

    Pure log-softmax → gather, with no masking or reduction. Both the
    pretrain and SFT loss paths layer masks and aggregations on top of
    this single per-token score, so the underlying arithmetic is shared
    without threading a ``reduction=`` flag through the public API.
    """
    log_probs = jax.nn.log_softmax(logits.astype(jnp.float32), axis=-1)
    log_p_x0 = jnp.take_along_axis(log_probs, x0[..., None], axis=-1).squeeze(-1)
    return -log_p_x0
