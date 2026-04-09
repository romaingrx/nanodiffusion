"""Named chat datasets for SFT.

Mirrors :mod:`nanodiffusion.data.datasets` but for indexable chat
conversation sources consumed by the SFT loader. HuggingFace-hosted
datasets go through :class:`HuggingFaceChatSource`; small non-HF
bundles (e.g. identity) register custom factories directly.

Adding a new chat dataset is one file under this package plus one
side-effect import below.
"""

# Side-effect imports: each module runs a ``register_hf_chat`` call (or
# hand-registers via ``CHAT_DATASETS.register``) at import time. The
# tuple reference keeps the static analyzer from flagging the modules
# as unused while staying honest about why the import block exists.
from nanodiffusion.data.chat_datasets import gsm8k, identity, smoltalk
from nanodiffusion.data.chat_datasets._base import (
    CHAT_DATASETS,
    ChatDatasetFactory,
    HuggingFaceChatSource,
    JsonlChatSource,
    RowToConversation,
    get_chat_dataset,
    normalize_message,
    register_hf_chat,
)

_SIDE_EFFECT_MODULES = (gsm8k, identity, smoltalk)

# Legacy decorator alias used by tests for one-off registrations.
register_chat = CHAT_DATASETS.register


__all__ = [
    "CHAT_DATASETS",
    "ChatDatasetFactory",
    "HuggingFaceChatSource",
    "JsonlChatSource",
    "RowToConversation",
    "get_chat_dataset",
    "normalize_message",
    "register_chat",
    "register_hf_chat",
]
