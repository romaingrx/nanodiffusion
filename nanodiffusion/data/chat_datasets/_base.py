"""Shared primitives for chat dataset factories.

Per-dataset modules declare a row decoder plus a single
``register_hf_chat`` call. Non-HF bundles hand-roll a factory and
call ``CHAT_DATASETS.register`` directly.
"""

import json
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from nanodiffusion.chat import Conversation, Message
from nanodiffusion.data.chat_source import ChatSource
from nanodiffusion.data.datasets import DownloadOptions
from nanodiffusion.registry import Registry

if TYPE_CHECKING:
    from datasets import Dataset

type RowToConversation = Callable[[dict[str, object]], Conversation]


@runtime_checkable
class ChatDatasetFactory(Protocol):
    """Builds a :class:`ChatSource` from a local cache directory.

    Parallel to :class:`nanodiffusion.data.datasets.DatasetFactory` so
    the same CLI shape drives both pretrain and SFT downloads.
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

    Schema-specific decoding lives in the caller-supplied
    ``row_to_conversation`` so this class stays dataset-agnostic.
    ``row_to_conversation`` must raise on undecodable rows rather
    than returning ``None`` — silent drops would cost training signal.
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

    Each line is a JSON array of ``{role, content}`` dicts matching
    :class:`Message`. For small non-HF bundles.
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

    Raises on missing fields or unsupported roles so a bad row aborts
    the load loudly instead of silently dropping training signal.
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
    """Register an HF-hosted chat dataset in one call."""

    def factory(
        data_dir: Path,
        *,
        download: bool = True,  # noqa: ARG001 -- datasets lib caches automatically
        download_options: DownloadOptions | None = None,  # noqa: ARG001
    ) -> ChatSource:
        # Deferred import: the ``datasets`` library is heavy and
        # listing the registry shouldn't pull it into process memory.
        from datasets import load_dataset  # noqa: PLC0415

        data_dir.mkdir(parents=True, exist_ok=True)
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
