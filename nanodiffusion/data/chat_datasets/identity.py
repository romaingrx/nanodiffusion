"""Identity conversations bundle (Karpathy S3).

Small JSONL hosted on a public S3 bucket outside the HuggingFace hub,
so this factory bypasses :func:`register_hf_chat` and goes through the
pretrain :func:`download_with_backoff` helper directly.
"""

from pathlib import Path

from nanodiffusion.data.chat_datasets._base import (
    CHAT_DATASETS,
    ChatDatasetFactory,
    JsonlChatSource,
)
from nanodiffusion.data.chat_source import ChatSource
from nanodiffusion.data.datasets import DownloadOptions, download_with_backoff

_URL = "https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl"
_FILENAME = "identity_conversations.jsonl"


@CHAT_DATASETS.register("identity")
def identity_factory(
    data_dir: Path,
    *,
    download: bool = True,
    download_options: DownloadOptions | None = None,
) -> ChatSource:
    """Identity conversations bundle from karpathy-public S3. ~1K rows, plain JSONL."""
    target = data_dir / _FILENAME
    if download and not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        download_with_backoff(
            _URL,
            target,
            options=download_options or DownloadOptions(),
        )
    return JsonlChatSource(target)


_: ChatDatasetFactory = identity_factory  # static protocol check
