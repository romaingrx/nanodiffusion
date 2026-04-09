import pytest

from nanodiffusion.chat import Conversation
from nanodiffusion.data.chat_source import (
    ChatSource,
    InMemoryChatSource,
    TaskMixture,
)


def _make_conv(content: str) -> Conversation:
    return {
        "messages": [
            {"role": "user", "content": content},
            {"role": "assistant", "content": f"echo: {content}"},
        ],
    }


def _conv_list(prefix: str, n: int) -> list[Conversation]:
    return [_make_conv(f"{prefix}-{i}") for i in range(n)]


def test_in_memory_chat_source_roundtrip() -> None:
    convs = _conv_list("a", 5)
    src = InMemoryChatSource(convs)
    assert len(src) == 5
    for i, original in enumerate(convs):
        assert src[i] == original


def test_in_memory_chat_source_is_runtime_checkable() -> None:
    """runtime_checkable is load-bearing for beartype's import-hook check."""
    src = InMemoryChatSource(_conv_list("a", 1))
    assert isinstance(src, ChatSource)


def test_in_memory_chat_source_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="at least one"):
        InMemoryChatSource([])


def test_task_mixture_length_is_sum_of_sources() -> None:
    a = InMemoryChatSource(_conv_list("a", 4))
    b = InMemoryChatSource(_conv_list("b", 3))
    mix = TaskMixture([a, b])
    assert len(mix) == 7


def test_task_mixture_shuffle_is_deterministic_with_seed() -> None:
    a = InMemoryChatSource(_conv_list("a", 5))
    b = InMemoryChatSource(_conv_list("b", 5))

    order1 = [mix[i] for mix in [TaskMixture([a, b], seed=123)] for i in range(10)]
    order2 = [mix[i] for mix in [TaskMixture([a, b], seed=123)] for i in range(10)]
    assert order1 == order2


def test_task_mixture_different_seeds_produce_different_order() -> None:
    a = InMemoryChatSource(_conv_list("a", 5))
    b = InMemoryChatSource(_conv_list("b", 5))

    mix1 = TaskMixture([a, b], seed=1)
    mix2 = TaskMixture([a, b], seed=2)
    order1 = [mix1[i] for i in range(10)]
    order2 = [mix2[i] for i in range(10)]
    assert order1 != order2


def test_task_mixture_oversamples_via_duplicate_entries() -> None:
    """Passing the same source twice doubles its contribution.

    Ports nanochat's ``tasks/common.py::TaskMixture`` oversampling trick:
    there's no separate multiplier — duplicates on the input list are
    the oversampling mechanism.
    """
    small = InMemoryChatSource(_conv_list("s", 2))
    big = InMemoryChatSource(_conv_list("b", 6))

    mix = TaskMixture([small, small, small, big], seed=42)
    assert len(mix) == 2 * 3 + 6

    # Count the appearance of each 'small' conversation across the mix.
    count = {"s-0": 0, "s-1": 0, "b-0": 0}
    for i in range(len(mix)):
        conv = mix[i]
        content = conv["messages"][0]["content"]
        if content in count:
            count[content] += 1
    assert count["s-0"] == 3
    assert count["s-1"] == 3
    assert count["b-0"] == 1


def test_task_mixture_rejects_empty_source_list() -> None:
    with pytest.raises(ValueError, match="at least one source"):
        TaskMixture([])


def test_task_mixture_rejects_empty_individual_source() -> None:
    """An empty constituent would make the final index_map shorter than
    advertised if we silently skipped it; catch it at construction."""

    class _EmptySource:
        def __len__(self) -> int:
            return 0

        def __getitem__(self, index: int) -> Conversation:  # pragma: no cover
            raise IndexError(index)

    with pytest.raises(ValueError, match="source 0 is empty"):
        TaskMixture([_EmptySource()])
