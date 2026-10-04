"""
Safe introspection of classes and type hints.

Versions of [`isinstance`][], [`issubclass`][], [`typing.get_origin`][]
and [`typing.get_args`][] that do not fail on hints, plus the helpers
that unwrap, normalise and convert hints.

The public names are re-exported by [`bagof.core.magic`][]; import them
from there.
"""

__all__ = [
    "is_typeddict",
    "typeddict_required_keys",
    "issubclassable",
    "safe_issubclass",
    "safe_isinstance",
    "normalise_hint",
    "unwrap",
    "safe_get_origin",
    "get_origin_uw",
    "safe_get_args",
    "get_args_uw",
    "get_concrete_type",
    "eq_safenan",
    "issubscriptable",
    "type2hint",
]

# stdlib
import collections
import contextlib
import inspect
import math
import re
import typing
from collections import abc

# dependencies
import typing_extensions as tx

# locals
from ._compat import REAL_TYPES, UNSET, NoneType, _is_special_form

# `typing.TypedDict` and `typing_extensions.TypedDict` are distinct
# objects on every Python this package supports, and a class built from
# one never mentions the other in its `__orig_bases__`. Both spellings
# describe the same thing, so treat them interchangeably throughout --
# `typing` is imported for this identity check alone, never for
# annotations (which go through `tx`, per the house style).
_TYPEDDICT_MARKERS = tuple(
    marker
    for marker in (tx.TypedDict, getattr(typing, "TypedDict", None))
    if marker is not None
)


def _is_typeddict_marker(cls: tx.Any) -> bool:
    """Whether `cls` is `TypedDict` itself, in either spelling."""
    return any(cls is marker for marker in _TYPEDDICT_MARKERS)


def _canonical_typeddict(cls: tx.Any) -> tx.Any:
    """Collapse either `TypedDict` spelling to the canonical one."""
    return tx.TypedDict if _is_typeddict_marker(cls) else cls


def is_typeddict(cls: tx.Any) -> bool:
    """
    Return true if an object is a [`TypedDict`][tx.TypedDict] or a subclass
    of it.

    !!! tip
        This function differs from
        [`typing.is_typeddict`][tx.is_typeddict] in that it returns `True`
        for [`TypedDict`][tx.TypedDict] itself.
    """
    if _is_typeddict_marker(cls):
        return True
    return tx.is_typeddict(cls)


def typeddict_required_keys(cls: tx.Any) -> tx.FrozenSet[str]:
    """
    The required keys of a [`TypedDict`][tx.TypedDict].

    Reads `__required_keys__` where the class has it -- the only source
    that accounts for [`Required`][typing.Required] /
    [`NotRequired`][typing.NotRequired] (in either nesting with
    [`Annotated`][typing.Annotated]) and for inheriting from bases
    declared with a different `total=`.

    Falls back to `__total__` where it does not:
    [`typing.TypedDict`][] gained `__required_keys__` only in Python 3.9,
    and before that a key's requiredness came from the class's `total=`
    alone -- per-key `Required`/`NotRequired` did not exist.

    !!! warning
        On older Pythons, a [`typing.TypedDict`][] that inherits from a
        base declared with a different `total=` reports **every**
        inherited key as required. The stdlib does not record which class
        declared a key, nor a usable link back to the base -- a subclass
        has no `__orig_bases__` and its `__mro__` reaches only
        [`dict`][] - so the true answer is not recoverable.

        The error is in the safe direction: a required key that is really
        optional makes a valid value fail loudly, rather than letting an
        invalid one through. Use
        [`typing_extensions.TypedDict`][tx.TypedDict], which reimplements
        the class precisely to fix this, when it matters.

    !!! example
        ```pycon
        >>> class Movie(TypedDict):
        ...     title: str
        ...     year: NotRequired[int]
        >>> typeddict_required_keys(Movie)
        frozenset({'title'})
        ```
    """
    keys = getattr(cls, "__required_keys__", None)
    if keys is not None:
        return frozenset(keys)
    annotations = getattr(cls, "__annotations__", {})
    if getattr(cls, "__total__", True):
        return frozenset(annotations)
    return frozenset()


def _all_orig_bases(cls: type, _self: bool = True) -> tx.Tuple[type, ...]:
    """Get all original bases of a type, including the type itself."""
    if not is_typeddict(cls):
        return ()
    bases = (cls,) if _self else ()
    for base in getattr(cls, '__orig_bases__', ()):
        if _is_typeddict_marker(base):
            # Appended once, canonically, at the end.
            continue
        bases += (base,) + _all_orig_bases(base, _self=False)
    if _self:
        # Always terminate with the canonical marker rather than trusting
        # `__orig_bases__` to contain one. `typing.TypedDict` records no
        # `__orig_bases__` at all on a sub-subclass, and the two spellings
        # never appear in each other's bases -- so deriving this from the
        # declared bases alone misses a typeddict that plainly is one.
        bases += (tx.TypedDict,)
    return bases


def issubclassable(cls: tx.Any) -> bool:
    """
    Return true if an object is a type or is [`TypedDict`][tx.TypedDict].

    !!! tip
        This function differs from `#!python isinstance(cls, type)` in that it
        returns [`True`][] for [`TypedDict`][tx.TypedDict] and its subclasses,
        even though they are not technically types.

    !!! note
        A typing construct - [`Any`][typing.Any], [`Union`][typing.Union],
        [`Literal`][typing.Literal] - is never subclassable, on any Python
        version. Some of them *are* classes on recent Pythons (`Any` from
        3.11, `Union` from 3.14), so `#!python isinstance(hint, type)`
        answers differently across the versions this package supports.
    """
    if _is_special_form(cls):
        return False
    if _is_typeddict_marker(cls):
        return True
    return isinstance(cls, type)


def safe_issubclass(subcls: tx.Any, cls: tx.Any) -> bool:
    """Safe subclass (does not fail if arguments are not types).

    !!! warning
        If `cls` is a [`TypedDict`][tx.TypedDict], this function looks
        at `subcls`'s `__orig_bases__`, instead of its `__bases__`.
        A plain [`dict`][] is *not* a subclass of a
        [`TypedDict`][tx.TypedDict] - the relation only holds the other
        way round.

    !!! example
        ```pycon
        >>> safe_issubclass(bool, int)
        True
        >>> safe_issubclass(bool, (str, int))  # a tuple, like `issubclass`
        True
        >>> safe_issubclass(int, "not a type")  # no error
        False
        ```
    """
    if isinstance(cls, tuple):
        return any(safe_issubclass(subcls, each) for each in cls)
    if is_typeddict(cls):
        return _canonical_typeddict(cls) in _all_orig_bases(subcls)
    if _is_special_form(cls) or _is_special_form(subcls):
        # A typing construct may be a real class on a recent Python
        # (`Any` from 3.11, `Union` from 3.14), so `issubclass` would
        # answer it - differently than on the versions before.
        return False
    if isinstance(subcls, type) and isinstance(cls, type):
        return issubclass(subcls, cls)
    return False


def safe_isinstance(obj: tx.Any, cls: tx.Any) -> bool:
    """
    Safe isinstance (does not fail if second argument is not a type).

    !!! warning
        A [`TypedDict`][tx.TypedDict] cannot be instance-checked. Python
        refuses `#!python isinstance(value, SomeTypedDict)` outright, and
        a TypedDict leaves no trace on the dict it describes, so there is
        nothing to recognise at runtime. This function therefore answers
        [`False`][] for one; validate the *shape* of the dict instead.

    !!! example
        ```pycon
        >>> safe_isinstance(1, int)
        True
        >>> safe_isinstance(1, (str, int))  # a tuple, like `isinstance`
        True
        >>> safe_isinstance(1, "not a type")  # no error
        False
        ```
    """
    if isinstance(cls, tuple):
        return any(safe_isinstance(obj, each) for each in cls)
    if is_typeddict(cls):
        return safe_issubclass(type(obj), cls)
    if isinstance(cls, type) and cls is not tx.Any:
        return isinstance(obj, cls)
    return False


def normalise_hint(hint: tx.Any) -> tx.Any:
    """
    Put a hint in its canonical form.

    A bare [`None`][] means [`NoneType`][types.NoneType] when it is used
    as a type hint, so it is replaced by it. Every other hint is returned
    unchanged.

    !!! note
        Only a bare `None` is replaced. A `None` *inside* a hint keeps its
        meaning: `#!python Literal[None]` is a literal `None` **value**,
        not a type.

    !!! example
        ```pycon
        >>> normalise_hint(None)
        <class 'NoneType'>
        >>> normalise_hint(int)
        <class 'int'>
        ```
    """
    return NoneType if hint is None else hint


def unwrap(hint: tx.Any, origin: tx.Any = (tx.Annotated,)) -> tx.Any:
    """
    Unwrap a type hint from its origin, if it is in the unwrap list.

    If [`TypeVar`][typing.TypeVar] is one of the origins to unwrap, it will
    be unwrapped to its default, its (union of) constraints, or its bound -
    in that order.

    !!! example
        ```pycon
        >>> from typing import Annotated
        >>> unwrap(Annotated[int, "meta"])
        <class 'int'>
        >>> unwrap(Annotated[Annotated[str, 1], 2])
        <class 'str'>
        >>> unwrap(int)  # unchanged
        <class 'int'>
        ```
    """
    if origin is None:
        origin = ()
    if isinstance(origin, str) or not isinstance(origin, abc.Sequence):
        # A `str` is a `Sequence`, but a single hint - not a list of them.
        origin = (origin,)
    if safe_get_origin(hint) in origin:
        return unwrap(tx.get_args(hint)[0], origin=origin)
    if tx.TypeVar in origin and safe_isinstance(hint, tx.TypeVar):
        return unwrap(_unwrap_typevar(hint), origin=origin)
    return hint


_unwrap = unwrap  # alias for convenience


def _unwrap_typevar(hint: tx.Any, __reentrant: tuple = ()) -> tx.Any:
    origin = get_origin_uw(hint)
    if origin in __reentrant:
        # A cycle (e.g. two typevars defaulting to each other). Returning
        # the typevar would send `unwrap` straight back in here, so answer
        # what an uninformative typevar answers.
        return tx.Any
    __reentrant += (origin,)
    if not safe_isinstance(origin, tx.TypeVar):
        return hint
    if getattr(origin, "__default__", tx.NoDefault) is not tx.NoDefault:
        return _unwrap_typevar(origin.__default__, __reentrant=__reentrant)
    if getattr(origin, "__constraints__", ()):
        return tx.Union[origin.__constraints__]
    if getattr(origin, "__bound__", None) is not None:
        return _unwrap_typevar(origin.__bound__, __reentrant=__reentrant)
    return tx.Any


def safe_get_origin(hint: tx.Any, unwrap: tx.Any = ()) -> tx.Any:
    """
    Safe version of [`tx.get_origin`][].

    Can also unwrap some hints (e.g. [`Annotated`][typing.Annotated])
    if asked.

    !!! note
        Unlike [`typing.get_origin`][], this returns the input hint
        itself, instead of `None`, when the hint is not a generic type.
    """
    if unwrap:
        hint = _unwrap(hint, origin=unwrap)
    origin = tx.get_origin(hint)
    if origin is None:
        return hint
    return origin


def get_origin_uw(hint: tx.Any) -> tx.Any:
    """
    Safe version of [`tx.get_origin`][] that unwraps
    [`Annotated`][typing.Annotated] hints.

    Returns the input type, instead of `None`, if the input is not a
    generic type.
    """
    return safe_get_origin(hint, unwrap=tx.Annotated)


def safe_get_args(hint: tx.Any, unwrap: tx.Any = ()) -> tx.Tuple[tx.Any, ...]:
    """
    Safe version of [`tx.get_args`][].

    Returns an empty tuple if the input is not a generic type.
    Can also unwrap some hints (e.g. [`Annotated`][typing.Annotated]) if asked.
    """
    hint = _unwrap(hint, origin=unwrap)
    return tx.get_args(hint)


def get_args_uw(hint: tx.Any) -> tx.Tuple[tx.Any, ...]:
    """
    Safe version of [`tx.get_args`][] that unwraps
    [`Annotated`][typing.Annotated] hints.

    Returns an empty tuple if the input is not a generic type.
    """
    return safe_get_args(hint, unwrap=tx.Annotated)


def get_concrete_type(hint: tx.Any, fallback: type = UNSET) -> tx.Type[tx.Any]:
    """
    Get a valid concrete type from a type hint.

    * If the hint is annotated, the [`Annotated`][typing.Annotated] wrapper
      is removed.
    * If the hint has an origin, it is used.
    * If the hint is a [`TypeVar`][typing.TypeVar]:
        - its default value is used, if it has one; otherwise
        - the **first** of its constraints is used, if it has any;
          otherwise
        - its bound is used, if it has one; otherwise
        - the fallback type is used, if it is provided; otherwise
        - a [`TypeError`][] is raised.
    * If the (resolved) hint is a concrete, non-abstract type, it is
      returned as is; otherwise
    * The fallback type is used, if it is provided; otherwise
    * A [`TypeError`][] is raised.

    !!! note
        A constrained typevar has no single concrete type - it stands for
        the union of its constraints - so the first constraint is taken,
        the same way [`get_default`][] takes the first value of a
        [`Literal`][tx.Literal].

    !!! example
        ```pycon
        >>> get_concrete_type(List[int])
        <class 'list'>
        >>> get_concrete_type(TypeVar("T", int, str))
        <class 'int'>
        ```
    """
    origin = safe_get_origin(hint, unwrap=(tx.Annotated, tx.TypeVar))
    if _is_concrete_type(origin):
        return origin
    concrete = _first_concrete_constraint(hint)
    if concrete is not None:
        return concrete
    if safe_isinstance(fallback, type):
        return fallback
    raise TypeError(
        f"Cannot get concrete type for hint {hint} (of type {type(hint)}) "
        f"and fallback {fallback} (of type {type(fallback)})."
    )


def _is_concrete_type(hint: tx.Any) -> bool:
    """Whether a hint is a class that can actually be instantiated."""
    if _is_special_form(hint):
        # `Union` is a class from python 3.14 on, but instantiating it
        # is still meaningless.
        return False
    return safe_isinstance(hint, type) and not inspect.isabstract(hint)


def _first_concrete_constraint(hint: tx.Any) -> tx.Optional[type]:
    """The first concrete constraint of a constrained typevar, if any."""
    typevar = unwrap(hint, tx.Annotated)
    if not safe_isinstance(typevar, tx.TypeVar):
        return None
    for constraint in getattr(typevar, "__constraints__", ()):
        origin = safe_get_origin(constraint, unwrap=(tx.Annotated,))
        if _is_concrete_type(origin):
            return origin
    return None


class _NaN:
    """The value every real NaN is mapped to by [`eq_safenan`][]."""

    def __repr__(self) -> str:
        return "<NaN>"


_NAN = _NaN()


def eq_safenan(x: tx.Any) -> tx.Any:
    """
    Map a value to a form that compares equal across NaNs.

    Since `#!python float("nan") != float("nan")`, comparing values that
    may contain NaN with `==` is unsafe. Apply this function to both
    operands before comparing them: real NaN values are all mapped to one
    sentinel (so that two NaNs compare equal), while every other value is
    returned unchanged.

    !!! note
        Only real numbers are recognised. A complex NaN is returned
        unchanged, and so still compares unequal to itself.

    !!! example
        ```pycon
        >>> nan = float("nan")
        >>> nan == nan
        False
        >>> eq_safenan(nan) == eq_safenan(nan)
        True
        ```
    """
    if isinstance(x, REAL_TYPES) and math.isnan(x):
        return _NAN
    return x


def issubscriptable(x: tx.Any) -> bool:
    """
    Check that an object is subscriptable (i.e. can be used with `[]`).

    True if the object is a type and has `__class_getitem__`, or if it
    is an instance and has `__getitem__`. Otherwise, returns False.
    """
    if isinstance(x, type) and hasattr(x, "__class_getitem__"):
        return True
    if not isinstance(x, type) and hasattr(x, "__getitem__"):
        return True
    return False


_TYPE2HINT_NAMES = (
    (dict, "Dict"),
    (frozenset, "FrozenSet"),
    (list, "List"),
    (set, "Set"),
    (tuple, "Tuple"),
    (type, "Type"),
    (abc.AsyncGenerator, "AsyncGenerator"),
    (abc.AsyncIterable, "AsyncIterable"),
    (abc.AsyncIterator, "AsyncIterator"),
    (abc.Awaitable, "Awaitable"),
    (abc.Callable, "Callable"),
    (abc.Collection, "Collection"),
    (abc.Container, "Container"),
    (abc.Coroutine, "Coroutine"),
    (abc.Generator, "Generator"),
    (abc.Hashable, "Hashable"),
    (abc.ItemsView, "ItemsView"),
    (abc.Iterable, "Iterable"),
    (abc.Iterator, "Iterator"),
    (abc.KeysView, "KeysView"),
    (abc.Mapping, "Mapping"),
    (abc.MappingView, "MappingView"),
    (abc.MutableMapping, "MutableMapping"),
    (abc.MutableSequence, "MutableSequence"),
    (abc.MutableSet, "MutableSet"),
    (abc.Reversible, "Reversible"),
    (abc.Sequence, "Sequence"),
    (abc.Set, "AbstractSet"),
    (abc.Sized, "Sized"),
    (abc.ValuesView, "ValuesView"),
    (collections.ChainMap, "ChainMap"),
    (collections.Counter, "Counter"),
    (collections.OrderedDict, "OrderedDict"),
    (collections.defaultdict, "DefaultDict"),
    (collections.deque, "Deque"),
    (contextlib.AbstractContextManager, "ContextManager"),
    (contextlib.AbstractAsyncContextManager, "AsyncContextManager"),
    (re.Match, "Match"),
    (re.Pattern, "Pattern"),
)
"""
The type hint each non-subscriptable type maps to, by name.

An explicit table, rather than deriving the name from the type's own: the
capitalisation does not follow (`defaultdict` becomes `DefaultDict`,
`frozenset` becomes `FrozenSet`, `abc.Set` becomes `AbstractSet`).
"""

_TYPE2HINT = {
    cls: getattr(tx, name)
    for cls, name in _TYPE2HINT_NAMES
    if hasattr(tx, name)
}
"""
[`_TYPE2HINT_NAMES`][], resolved against the running `typing_extensions`.

Entries whose hint the running version does not provide (`ByteString`
was removed, for example) are simply left out, so the type is returned
unchanged rather than raising at import time.
"""


def type2hint(x: tx.Any) -> tx.Any:
    """
    Convert a type to a (subscriptable) type hint.

    * If the input is a type, and it does not have `__class_getitem__`,
      we try to find its corresponding type hint.
      For example, in python 3.8, `#!python type2hint(list)` returns
      [`typing.List`][tx.List].
    * Otherwise, the value is returned as is.

    !!! example
        ```pycon
        >>> type2hint(frozenset)
        typing.FrozenSet
        >>> type2hint(3)  # not a type, so unchanged
        3
        ```
    """
    if issubscriptable(x):
        return x
    try:
        return _TYPE2HINT.get(x, x)
    except TypeError:
        # Unhashable: cannot be a key, so there is nothing to look up.
        return x
