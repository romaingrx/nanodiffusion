"""Named chat datasets for SFT.

Adding a new chat dataset is one file under this package plus one
side-effect import below.
"""

# Side-effect imports: each module registers itself at import time.
# The tuple below anchors the imports against an unused-import warning.
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
