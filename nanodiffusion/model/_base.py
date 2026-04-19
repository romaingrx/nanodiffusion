"""Base class for masked discrete diffusion models.

Kept in its own file so concrete models can import :class:`DiffusionModel`
without triggering a circular import via ``nanodiffusion.model.__init__``.
"""

import equinox as eqx

from nanodiffusion.types import Logits, Scalar, Tokens


class DiffusionModel(eqx.Module):
    """Nominal base for masked discrete diffusion models.

    Subclasses override :meth:`__call__` to map a token sequence and a
    diffusion-time scalar to per-position vocabulary logits. The class
    itself declares no fields; it exists so the training loop, loss,
    sampler and checkpoint API can bind generics to a single tight
    ``eqx.Module`` subtype without leaking equinox into their public
    signatures.
    """

    def __call__(self, tokens: Tokens, t: Scalar) -> Logits:
        msg = f"{type(self).__name__} must override __call__"
        raise NotImplementedError(msg)
