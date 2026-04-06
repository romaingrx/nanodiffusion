from typing import Protocol, runtime_checkable

from nanodiffusion.model.transformer import Transformer
from nanodiffusion.types import Logits, Scalar, Tokens

__all__ = ["DiffusionModel", "Transformer"]


@runtime_checkable
class DiffusionModel(Protocol):
    def __call__(self, tokens: Tokens, t: Scalar) -> Logits: ...
