"""Generic named-factory registry with duplicate-guard on registration."""

from collections.abc import Callable, Iterator, MutableMapping


class Registry[T](MutableMapping[str, T]):
    """Named-factory store.

    ``kind`` is the human-readable noun used in error messages so
    different registries ("dataset", "chat dataset", ...) surface
    distinct lookup failures. Inheriting from :class:`MutableMapping`
    keeps ``monkeypatch.setitem`` and dict-style test idioms working.
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

        The guard lives here rather than in ``__setitem__`` so
        ``monkeypatch.setitem`` can still override for the lifetime of
        a test.
        """

        def decorator(fn: T) -> T:
            if name in self._items:
                msg = f"{self._kind} {name!r} already registered"
                raise ValueError(msg)
            self._items[name] = fn
            return fn

        return decorator
