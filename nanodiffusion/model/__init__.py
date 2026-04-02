from typing import Protocol

from nanodiffusion.model.transformer import Transformer
from nanodiffusion.types import Logits, Scalar, Tokens

__all__ = ["DiffusionModel", "Transformer"]


class DiffusionModel(Protocol):
    def __call__(self, tokens: Tokens, t: Scalar) -> Logits: ...
