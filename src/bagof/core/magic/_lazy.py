"""
Lazy registry keys and lazy class attributes, for optional libraries.

A registry (a dict searched by
[`get_from_registry`][bagof.core.magic.get_from_registry]) can be keyed
by a [`ForwardRef`][typing.ForwardRef] instead of the object itself, so
that registering a value for a type from an optional library does not
import that library. The key is kept pending and moved into the
registry - keyed by the object it names - once someone else has imported
it. Nothing here ever imports a module or evaluates a string to resolve
one.

The public names are re-exported by [`bagof.core.magic`][]; import them
from there.
"""

__all__ = [
    "clear_pending",
    "defer",
    "find_name",
    "has_module",
    "is_forward_ref",
    "lazy_import",
    "pending",
    "resolve_pending",
]

# stdlib
import importlib
import importlib.util
import sys
import threading
import types
import typing
import warnings
import weakref

# dependencies
import typing_extensions as tx

_FORWARD_REFS = tuple({typing.ForwardRef, tx.ForwardRef})
"""The forward-reference types that make a lazy registry key."""

_MISSING = object()
"""Marker for a name that cannot be resolved (yet)."""


class _Pending(tx.NamedTuple):
    """A registration whose forward-reference key is not resolved yet."""

    registry: tx.Callable[[], tx.Optional[dict]]
    """Returns the registry, or `None` once it has been collected."""
    context: tx.Optional[str]
    """The module the name is relative to."""
    name: str
    """The dotted name the forward reference holds."""
    value: tx.Any
    """The value to register."""


_PENDING: tx.List[_Pending] = []
"""
The registrations whose key is not resolved yet, oldest first. A hint of a
type from a module that was never imported cannot reach a registry, so
nothing is lost by waiting.
"""

_PENDING_LOCK = threading.RLock()
"""
Guards every read-modify-write of `_PENDING`. Registry writes happen
after it is released, since they may run arbitrary code (`__hash__`,
`__eq__`, a `__setitem__` override).
"""


def is_forward_ref(obj: tx.Any) -> bool:
    """Whether `obj` is a [`ForwardRef`][typing.ForwardRef] (a lazy key)."""
    return isinstance(obj, _FORWARD_REFS)


def _registry_ref(registry: dict) -> tx.Callable[[], tx.Optional[dict]]:
    """A weak reference to `registry` if it takes one, else a strong one."""
    try:
        return weakref.ref(registry)
    except TypeError:
        # A plain `dict` cannot be weakly referenced (a subclass can).
        return lambda: registry


def defer(
    registry: dict,
    ref: tx.Any,
    value: tx.Any,
    context: tx.Optional[str] = None,
) -> None:
    """
    Register `value` in `registry` under the object `ref` names, once that
    object has been imported - see [`find_name`][] for how it is found.

    This is the lazy counterpart of `#!python registry[key] = value`: if
    the object can already be found, it is registered at once; otherwise
    the registration is kept pending, and [`resolve_pending`][] (which
    [`get_from_registry`][bagof.core.magic.get_from_registry] calls
    before every lookup) moves it into the registry as soon as the object
    can be found. Nothing is ever imported
    or evaluated to find it.

    !!! note "Names that only exist for type checkers"
        A name imported under `#!python if TYPE_CHECKING:` does not exist
        at runtime, so a key spelled with it never resolves. Use the real
        dotted path - `ForwardRef("dask.array.Array")`, not an alias such
        as `"da.Array"` - or `ForwardRef("Array", module="dask.array")`.
        Builtins are not looked up either: `ForwardRef("int")` never
        resolves (key the registry by `int` itself). [`pending`][] lists
        the keys still waiting.

    !!! note "Ordering"
        "Last registration wins" holds as for plain keys:

        - a newer registration of the same name in the same registry
          replaces an older pending one;
        - pending entries are applied oldest first, and an entry that
          resolves now is registered at once;
        - a pending entry never overrides a key already present in the
          registry when it resolves. It is pending only because its
          object could not be reached when it was registered, so a key
          already there for that object was (almost always) written
          later - typically a plain `#!python registry[obj] = value`.

        Consequently, two different spellings of one object that resolve
        in different passes are not ordered by registration: the first
        one to resolve wins. (Entries resolving in the same pass keep
        registration order.)

    !!! note "Lifetime"
        A pending entry holds a weak reference to `registry` when it
        supports one (a `dict` subclass), and is dropped once the registry
        is collected; a plain `dict` is held strongly until the entry
        resolves, or until [`clear_pending`][]. `value` is always held
        strongly.

    Parameters
    ----------
    registry : dict
        The registry to fill.
    ref : ForwardRef
        The forward reference. It must hold a dotted name
        (`"package.module.Name"`, `"Name"`), not an expression such as
        `"Optional[Name]"`. Its `module=` (Python 3.9.7+), if set, takes
        precedence over `context`.
    value : Any
        The value to register.
    context : str, optional
        The module a relative name is looked up in, usually the
        `__module__` of the class being registered.

    Raises
    ------
    TypeError
        If `ref` is not a [`ForwardRef`][typing.ForwardRef].
    ValueError
        If `ref` does not hold a dotted name.
    """
    if not is_forward_ref(ref):
        raise TypeError(f"Expected a ForwardRef, got {ref!r}")
    name = ref.__forward_arg__
    if not all(part.isidentifier() for part in name.split(".")):
        raise ValueError(
            f"A lazy registry key must be a dotted name, got {name!r}"
        )
    context = getattr(ref, "__forward_module__", None) or context
    # Settle the older entries that are resolvable first, so that they do
    # not override this newer one when they are applied later.
    resolve_pending()
    obj = _find_name(name, context)
    with _PENDING_LOCK:
        # Compare names, not refs: `ForwardRef` equality also involves
        # fields that vary across Python versions.
        _PENDING[:] = [
            entry for entry in _PENDING
            if entry.registry() is not None and not (
                entry.registry() is registry
                and (entry.context, entry.name) == (context, name)
            )
        ]
        if obj is _MISSING:
            _PENDING.append(
                _Pending(_registry_ref(registry), context, name, value)
            )
    if obj is not _MISSING:
        # Unconditional, like a plain key: this is the newest
        # registration.
        registry[obj] = value


def resolve_pending() -> None:
    """
    Move the pending registrations that can now be resolved into their
    registry, oldest first. Returns at once when nothing is pending.

    [`get_from_registry`][bagof.core.magic.get_from_registry] calls this
    before every lookup, so a caller only needs it before reading a
    registry directly. A resolved entry is not written over a key already
    present for its object - see the ordering note of [`defer`][].

    !!! note "Cost"
        Every pending entry is looked up on each call, so a lookup costs
        time linear in the number of entries that never resolve (about a
        microsecond each - an optional library that is installed but
        never imported, or a misspelt name).

    !!! note
        A name that resolves to an unhashable object cannot key a
        registry: its entry is dropped with a [`RuntimeWarning`][].
    """
    if not _PENDING:
        return
    with _PENDING_LOCK:
        snapshot = list(_PENDING)
    # Look the names up outside the lock: walking a class attribute may
    # run arbitrary code (descriptors, metaclass `__getattr__`).
    found = []
    for entry in snapshot:
        if entry.registry() is None:
            found.append((entry, _MISSING))
            continue
        obj = _find_name(entry.name, entry.context)
        if obj is not _MISSING:
            found.append((entry, obj))
    if not found:
        return
    # Pop under the lock, so that each entry is applied at most once.
    # Identity: another thread may have resolved this entry, or replaced
    # it with a newer registration, in the meantime.
    with _PENDING_LOCK:
        live = {id(entry) for entry in _PENDING}
        found = [(entry, obj) for entry, obj in found if id(entry) in live]
        popped = {id(entry) for entry, _ in found}
        _PENDING[:] = [e for e in _PENDING if id(e) not in popped]
    # Write after releasing it: a registry write may run arbitrary code.
    written: tx.List[tx.Tuple[dict, tx.Any]] = []
    for entry, obj in found:
        registry = entry.registry()
        if registry is None or obj is _MISSING:
            continue
        try:
            if obj in registry and not any(
                reg is registry and key is obj for reg, key in written
            ):
                # A newer (plain) key for this object: keep it.
                continue
            registry[obj] = entry.value
        except TypeError:
            warnings.warn(
                f"Lazy registry key {entry.name!r} resolved to an "
                f"unhashable object {obj!r}; dropped.",
                RuntimeWarning,
                stacklevel=3,
            )
            continue
        written.append((registry, obj))


def pending(
    registry: tx.Optional[dict] = None,
) -> tx.List[tx.Tuple[tx.Optional[str], str, tx.Any]]:
    """
    The registrations still pending (in `registry`, or in any registry),
    oldest first, as `(context_module, "dotted.name", value)`. Meant for
    debugging a key that never resolves.
    """
    with _PENDING_LOCK:
        entries = list(_PENDING)
    out = []
    for entry in entries:
        target = entry.registry()
        if target is None:
            continue
        if registry is None or target is registry:
            out.append((entry.context, entry.name, entry.value))
    return out


def clear_pending(registry: tx.Optional[dict] = None) -> None:
    """
    Drop the pending registrations (of `registry`, or of every registry)
    without applying them. Meant for test isolation.
    """
    with _PENDING_LOCK:
        _PENDING[:] = [
            entry for entry in _PENDING
            if registry is not None
            and entry.registry() is not None
            and entry.registry() is not registry
        ]


def find_name(name: str, context: tx.Optional[str] = None) -> tx.Any:
    """
    The object a dotted name refers to, if it is already imported. Never
    imports anything, and never evaluates the name.

    1. **Relative:** the first part of the name is looked up in the
       globals of the `context` module (if it is imported), and the rest
       is walked as attributes. This follows the module's own runtime
       imports, such as `#!python import numpy as np` for `"np.ndarray"`.
    2. **Absolute:** the longest dotted prefix of the name present in
       [`sys.modules`][] is taken, and the rest is walked as attributes.

    A module is always read through its `__dict__`, so a module-level
    `__getattr__` ([PEP 562](https://peps.python.org/pep-0562/)) cannot
    import anything: a submodule that has not been imported yet is not
    found. Builtins are not looked up.

    Returns
    -------
    Any
        The object.

    Raises
    ------
    NameError
        If the name cannot be found (yet).
    """
    obj = _find_name(name, context)
    if obj is _MISSING:
        raise NameError(
            f"{name!r} cannot be found"
            + (f" from {context!r}" if context else "")
            + " among the imported modules"
        )
    return obj


def _find_name(name: str, context: tx.Optional[str] = None) -> tx.Any:
    """[`find_name`][], returning `_MISSING` instead of raising."""
    parts = name.split(".")
    # 1. relative to the context module
    module = sys.modules.get(context) if context else None
    if module is not None:
        obj = _walk(module, parts)
        if obj is not _MISSING:
            return obj
    # 2. absolute
    for i in range(len(parts), 0, -1):
        module = sys.modules.get(".".join(parts[:i]))
        if module is not None:
            return _walk(module, parts[i:])
    return _MISSING


def _walk(obj: tx.Any, attrs: tx.Iterable[str]) -> tx.Any:
    """Walk attributes without triggering a module-level `__getattr__`."""
    for attr in attrs:
        if isinstance(obj, types.ModuleType):
            obj = vars(obj).get(attr, _MISSING)
        else:
            obj = getattr(obj, attr, _MISSING)
        if obj is _MISSING:
            # Not imported yet, or still initialising.
            break
    return obj


def has_module(name: str) -> bool:
    """
    Whether the top-level package of `name` can be imported, without
    importing it.

    Only the top-level name is looked up - finding a submodule's spec
    would import its parent package - so `#!python has_module("dask.array")`
    tells whether `dask` is installed.
    """
    try:
        return importlib.util.find_spec(name.partition(".")[0]) is not None
    except (ImportError, ValueError):  # pragma: no cover
        return False


def _import_name(name: str) -> tx.Any:
    """Import the object named by a fully-qualified dotted name."""
    first, *rest = name.split(".")
    obj = importlib.import_module(first)
    for attr in rest:
        try:
            obj = getattr(obj, attr)
        except AttributeError:
            if not isinstance(obj, types.ModuleType):
                raise
            # A submodule that has not been imported yet.
            obj = importlib.import_module(f"{obj.__name__}.{attr}")
    return obj


class lazy_import:
    """
    Class attribute holding an object that is only imported on first
    access, so that defining a class for an optional library does not
    import it.

    The name must be fully qualified (`"package.module.Name"`); it is
    imported the first time the attribute is read, and kept.

    !!! warning
        *Any* read of the class attribute imports it - including
        [`dir`][]-and-`getattr` walks such as [`inspect.getmembers`][],
        documentation generators, or `cls.DEFAULT` used as a fallback
        registry key. Pass explicit (lazy) keys where that matters.

    !!! example
        ```python
        class ToDaskArray(ArrayConverter):
            DEFAULT = lazy_import("dask.array.Array")
        ```
    """

    def __init__(self, name: str) -> None:
        self.name = name

    def __get__(self, obj: tx.Any, owner: tx.Any = None) -> tx.Any:
        # Importing twice (two threads racing here) is harmless: the
        # import system hands both the same object.
        if "value" not in self.__dict__:
            self.value = _import_name(self.name)
        return self.value

    def __repr__(self) -> str:
        return f"lazy_import({self.name!r})"
