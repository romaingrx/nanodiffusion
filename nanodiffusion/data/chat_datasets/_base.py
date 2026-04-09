"""Shared primitives for chat dataset factories.

Per-dataset modules (``smoltalk.py``, ``gsm8k.py``, ``identity.py``, ...)
live alongside this file and only declare their row decoder + a single
``register_hf_chat`` call. HuggingFace-hosted datasets go through
:class:`HuggingFaceChatSource`; anything else (e.g. the identity JSONL
bundle on S3) can hand-roll its own factory and call
``CHAT_DATASETS.register`` directly.
"""

import json
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from nanodiffusion.chat import Conversation, Message
from nanodiffusion.data.chat_source import ChatSource
from nanodiffusion.data.datasets import DownloadOptions
from nanodiffusion.data.registry import Registry

if TYPE_CHECKING:
    from datasets import Dataset

type RowToConversation = Callable[[dict[str, object]], Conversation]


@runtime_checkable
class ChatDatasetFactory(Protocol):
    """Builds a :class:`ChatSource` from a local cache directory.

    Mirrors :class:`nanodiffusion.data.datasets.DatasetFactory` — same
    ``(data_dir, download, download_options)`` triplet — so the same
    CLI shape can drive both pretrain and SFT downloads.
    """

    def __call__(
        self,
        data_dir: Path,
        *,
        download: bool = True,
        download_options: DownloadOptions | None = None,
    ) -> ChatSource: ...


CHAT_DATASETS: Registry[ChatDatasetFactory] = Registry("chat dataset")


def get_chat_dataset(name: str) -> ChatDatasetFactory:
    """Look up a registered chat dataset factory by name."""
    return CHAT_DATASETS[name]


class HuggingFaceChatSource:
    """Wraps a :class:`datasets.Dataset` as a :class:`ChatSource`.

    Schema-specific decoding — e.g. GSM8K's tool-call flattening — lives
    in a caller-supplied ``row_to_conversation`` callable so this class
    stays dataset-agnostic. The wrapped ``Dataset`` handles caching,
    random access, and the underlying parquet read so neither this class
    nor the chat-loader has to reimplement that machinery.

    ``row_to_conversation`` must not return ``None``: rows that cannot be
    decoded should raise so misconfigured datasets fail loudly at load
    time rather than silently dropping training signal.
    """

    def __init__(
        self,
        dataset: "Dataset",
        row_to_conversation: RowToConversation,
    ) -> None:
        if len(dataset) == 0:
            err = "HuggingFaceChatSource wraps an empty dataset"
            raise ValueError(err)
        self._dataset = dataset
        self._row_to_conversation = row_to_conversation

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index: int) -> Conversation:
        return self._row_to_conversation(self._dataset[index])


class JsonlChatSource:
    """Reads one conversation per line from a JSONL file.

    Each line must be a JSON array of ``{role, content}`` dicts matching
    :class:`Message`. Used for small bundles (e.g. the identity
    conversations) that aren't on the HuggingFace hub.
    """

    def __init__(self, path: Path) -> None:
        if not path.exists():
            err = f"JsonlChatSource file not found: {path}"
            raise FileNotFoundError(err)
        conversations: list[Conversation] = []
        with path.open(encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line:
                    continue
                messages = json.loads(line)
                if not isinstance(messages, list):
                    err = f"JsonlChatSource expected list, got {type(messages)}"
                    raise TypeError(err)
                conversations.append({"messages": messages})
        if not conversations:
            err = f"JsonlChatSource file is empty: {path}"
            raise ValueError(err)
        self._conversations = conversations

    def __len__(self) -> int:
        return len(self._conversations)

    def __getitem__(self, index: int) -> Conversation:
        return self._conversations[index]


def normalize_message(raw: object) -> Message:
    """Coerce a dataset row entry into a :class:`Message`.

    Raises on missing fields or unsupported roles rather than returning
    ``None`` — a single bad row should abort the load so the caller
    notices and fixes the schema drift instead of training on a silently
    truncated dataset. The ``match`` on role narrows to the
    :data:`~nanodiffusion.chat.Role` literal so the return dict
    satisfies the :class:`Message` TypedDict without a cast.
    """
    if not isinstance(raw, dict):
        err = f"Expected dict message, got {type(raw).__name__}"
        raise TypeError(err)
    role = raw.get("role")
    content = raw.get("content")
    if not isinstance(content, str):
        err = f"Message has bad content type: {raw!r}"
        raise TypeError(err)
    match role:
        case "user" | "assistant" | "system":
            return {"role": role, "content": content}
        case _:
            err = f"Message has unsupported role {role!r}"
            raise ValueError(err)


def register_hf_chat(
    *,
    name: str,
    repo_id: str,
    subset: str | None,
    split: str,
    row_to_conversation: RowToConversation,
    doc: str,
) -> ChatDatasetFactory:
    """Register an HF-hosted chat dataset in one call.

    Per-dataset modules declare their spec as a single call to this
    helper plus the row decoder it references; the ``datasets`` library
    is imported lazily inside the factory so just listing the registry
    (e.g. ``data list-chat``) doesn't pull the heavy dependency into
    process memory.
    """

    def factory(
        data_dir: Path,
        *,
        download: bool = True,  # noqa: ARG001 -- datasets lib caches automatically
        download_options: DownloadOptions | None = None,  # noqa: ARG001
    ) -> ChatSource:
        # The ``datasets`` library is heavy — defer the import so that
        # just listing the registry doesn't pull it into process memory.
        from datasets import load_dataset  # noqa: PLC0415

        data_dir.mkdir(parents=True, exist_ok=True)
        # Passing ``split`` narrows ``load_dataset``'s return to
        # ``Dataset`` in the stubs, so :class:`HuggingFaceChatSource`
        # accepts it directly without a cast or isinstance check.
        dataset = load_dataset(
            repo_id,
            subset,
            split=split,
            cache_dir=str(data_dir),
        )
        return HuggingFaceChatSource(dataset, row_to_conversation)

    factory.__doc__ = doc
    CHAT_DATASETS.register(name)(factory)
    return factory
