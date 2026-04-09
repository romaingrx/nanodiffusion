"""Generic named-factory registry.

Shared by pretrain datasets and SFT chat datasets so both can use the
same register/get/iterate surface without copy-pasting the glue. Inherits
from :class:`collections.abc.MutableMapping` so ``monkeypatch.setitem``
and the tests' ``.pop`` / ``.update`` / ``.clear`` idioms keep working
unchanged.
"""

from collections.abc import Callable, Iterator, MutableMapping


class Registry[T](MutableMapping[str, T]):
    """Named-factory store with duplicate-guard on ``register``.

    ``kind`` is the human-readable noun that prefixes error messages
    ("Unknown dataset 'foo'" / "Unknown chat dataset 'foo'") so the two
    registries surface distinct errors without each maintaining its own
    formatting code.
    """

    def __init__(self, kind: str) -> None:
        self._kind = kind
        self._items: dict[str, T] = {}

    def __getitem__(self, name: str) -> T:
        if name not in self._items:
            available = ", ".join(sorted(self._items)) or "(none)"
            msg = f"Unknown {self._kind} {name!r}. Available: {available}"
            raise KeyError(msg)
        return self._items[name]

    def __setitem__(self, name: str, value: T) -> None:
        self._items[name] = value

    def __delitem__(self, name: str) -> None:
        del self._items[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def register(self, name: str) -> Callable[[T], T]:
        """Decorator adding ``fn`` under ``name``; raises on duplicate.

        The duplicate-guard lives here, not in ``__setitem__``, so that
        ``monkeypatch.setitem(DATASETS, "foo", fake)`` can still override
        a registered factory for the lifetime of a test.
        """

        def decorator(fn: T) -> T:
            if name in self._items:
                msg = f"{self._kind} {name!r} already registered"
                raise ValueError(msg)
            self._items[name] = fn
            return fn

        return decorator
