
__all__ = [
    "MagicError",
    "MagicHint",
    "MultipleCauses",
    "eq_safenan",
    "get_concrete_type",
    "get_default",
    "get_from_registry",
    "get_origin_uw",
    "get_args_uw",
    "safe_get_origin",
    "safe_get_args",
    "safe_isinstance",
    "safe_issubclass",
    "ishintstance",
    "issubhint",
    "issubclassable",
    "issubscriptable",
    "is_typeddict",
    "typeddict_required_keys",
    "type2hint",
    "unwrap",
    "Unset",
    "UNSET",
    "NoneType",
    "REAL_TYPES",
    "UNION_TYPES",
    "UnionType",
]

# stdlib
import collections
import copy
import inspect
import math
import numbers
import typing
from collections import abc

# dependencies
import typing_extensions as tx

# optionals
if tx.TYPE_CHECKING:
    from types import NoneType, UnionType

    import numpy as _np
else:
    try:
        from types import NoneType, UnionType
    except ImportError:  # pragma: no cover  -- Python < 3.10
        NoneType = type(None)
        UnionType = tx.Union

    try:
        import numpy as _np
    except ImportError:  # pragma: no cover  -- numpy is optional
        _np = None

# typing
T = tx.TypeVar("T", covariant=True)

# constants
UNION_TYPES = (
    (tx.Union,) if UnionType is tx.Union else (tx.Union, UnionType)
)
"""The union spellings this package understands."""

REAL_TYPES = (
    (numbers.Real, _np.floating) if _np is not None else (numbers.Real,)
)
"""The real-number types [`eq_safenan`][] recognises."""

_SPECIAL_FORMS = (tx.Any, tx.Optional, tx.Literal) + UNION_TYPES
"""
The typing constructs that must never be treated as classes.

Several of these *are* classes on some Python versions and not on others
- [`Any`][typing.Any] became one in 3.11, and [`Union`][typing.Union]
became one in 3.14, when it merged with
[`types.UnionType`][] - so `#!python isinstance(hint, type)` silently
gives different answers across the versions this package supports. Pin
the answer instead of inheriting it.
"""


def _is_special_form(hint: tx.Any) -> bool:
    """Whether a hint is a typing construct rather than a class."""
    # Identity, not `in`: `==` on typing objects can be surprising.
    return any(hint is form for form in _SPECIAL_FORMS)


class Unset:

    def __new__(cls, *args, **kwargs) -> tx.Self:
        # `cls.__dict__`, not `hasattr`: the latter finds an inherited
        # `_INSTANCE`, so a subclass would hand back the base's instance.
        if "_INSTANCE" not in cls.__dict__:
            cls._INSTANCE = object.__new__(cls)
        return cls._INSTANCE

    def __bool__(self) -> bool:
        return False

    def __repr__(self) -> str:
        return "<UNSET>"

    def __str__(self) -> str:
        return "<UNSET>"


UNSET = Unset()
"""
A value that indicates that an argument was not set.

!!! note
    This is different from [`None`][], which may be a valid value.
"""


class MagicHint(tx.Generic[T]):
    """Base class for magic objects (factories, converters)."""

    BOUND = tx.Any
    """
    The type hint that this magic object is bound to.
    """

    DEFAULT = tx.Any
    """
    The default type hint for this magic object.
    """

    FALLBACK = UNSET
    """
    A concrete fallback type, used when the type hint does not resolve to
    a concrete class - for example, when it is an abstract class, or a
    bare typing construct (such as [`Union`][typing.Union] or
    [`Literal`][typing.Literal]) with no concrete origin of its own.
    """

    UNWRAP: tx.Tuple[tx.Any, ...] = (tx.Annotated, tx.TypeVar)
    """
    The hints that [`unwrapped`][], [`origin`][] and [`args`][] transparently
    unwrap before introspecting [`hint`][].

    !!! note
        A [`TypeVar`][typing.TypeVar] is resolved to its default, its
        (union of) constraints, or its bound - in that order - so that a
        typevar is introspected exactly like the hint it stands for. This
        matches [`fallback`][], which resolves typevars through
        [`get_concrete_type`][].

    Set to `(tx.Annotated,)` to opt out and introspect typevars as-is.
    """

    _FROZEN = ("hint",)
    """The attributes that cannot be reassigned after construction."""

    _CACHED: tx.Tuple[str, ...] = (
        "_unwrapped", "_origin", "_args", "_fallback"
    )
    """
    The attributes that memoise something derived from [`hint`][].

    [`rebind`][] clears these. A subclass that memoises more should
    extend this tuple rather than replace it.
    """

    def __init__(self, hint: tx.Any = UNSET) -> None:
        """
        Parameters
        ----------
        hint : Any, optional
            The type hint to use for this magic object.
            If not provided, the default hint for the class is used.

        !!! note
            [`hint`][] cannot be reassigned afterwards - the introspected
            properties are computed once and kept. Use [`rebind`][] to
            get a copy that describes a different hint.
        """
        # Recorded before the default is substituted, so that "no hint was
        # given" stays distinguishable from "a hint equal to `DEFAULT` was
        # given". `Annotated` metadata relies on the difference.
        self._hint_given = hint is not UNSET
        if hint is UNSET:
            hint = self.DEFAULT
        self.hint = normalise_hint(hint)
        self.__post_init__()

    @property
    def has_explicit_hint(self) -> bool:
        """
        Whether a hint was passed to the constructor.

        `False` when the object fell back to its [`DEFAULT`][] - which is
        not the same as carrying a hint that happens to equal it.
        """
        return getattr(self, "_hint_given", True)

    def rebind(self, hint: tx.Any) -> tx.Self:
        """
        Return a copy of this object describing a different hint.

        Every other attribute is carried over, so a configured object
        keeps its configuration - a threshold, a pattern, a length. The
        memoised properties listed in [`_CACHED`][] are recomputed.

        !!! example
            ```pycon
            >>> validator = IsGreaterThan(0)      # hint defaults to Number
            >>> stricter = validator.rebind(int)
            >>> stricter.threshold, stricter.hint
            (0, <class 'int'>)
            ```
        """
        new = copy.copy(self)
        for name in self._CACHED:
            new.__dict__.pop(name, None)
        # `hint` is frozen, so assign through `__dict__` rather than
        # tripping the guard that exists to stop exactly this happening
        # to a *live* object. This one is a fresh copy.
        new.__dict__["hint"] = normalise_hint(hint)
        new.__dict__["_hint_given"] = True
        new.__post_init__()
        return new

    def __setattr__(self, name: str, value: tx.Any) -> None:
        # `unwrapped`/`origin`/`args`/`fallback` are computed once and
        # kept, and `__post_init__` runs once - so a reassigned `hint`
        # would leave the object describing the hint it no longer has,
        # and would skip its own `BOUND` check. Refuse instead.
        if name in self._FROZEN and name in self.__dict__:
            raise AttributeError(
                f"{type(self).__name__}.{name} cannot be reassigned; "
                f"build a new {type(self).__name__} instead"
            )
        super().__setattr__(name, value)

    def __post_init__(self) -> None:
        if not issubhint(self.hint, self.BOUND):
            raise TypeError(
                f"Hint {self.hint} is not a valid subhint for {self.BOUND}"
            )

    @property
    def unwrapped(self) -> tx.Any:
        """
        The unwrapped type hint, with any hint listed in [`UNWRAP`][]
        (by default, [`Annotated`][typing.Annotated] wrappers and
        [`TypeVar`][typing.TypeVar]s) removed.
        """
        if getattr(self, "_unwrapped", None) is None:
            self._unwrapped = self._get_unwrapped()
        return self._unwrapped

    def _get_unwrapped(self) -> tx.Any:
        return unwrap(self.hint, self.UNWRAP)

    @property
    def origin(self) -> tx.Any:
        """
        The "safe" origin of the type hint

        * Any hint listed in [`UNWRAP`][] is removed (by default,
          [`Annotated`][typing.Annotated] wrappers and
          [`TypeVar`][typing.TypeVar]s).
        * If the origin is [`None`][], the hint itself is returned.
        """
        if getattr(self, "_origin", None) is None:
            self._origin = self._get_origin()
        return self._origin

    def _get_origin(self) -> tx.Any:
        return safe_get_origin(self.hint, unwrap=self.UNWRAP)

    @property
    def args(self) -> tx.Tuple[tx.Any, ...]:
        """
        The "safe" arguments of the type hint

        * Any hint listed in [`UNWRAP`][] is removed (by default,
          [`Annotated`][typing.Annotated] wrappers and
          [`TypeVar`][typing.TypeVar]s).
        * If the origin is [`None`][], returns an empty tuple.
        """
        if getattr(self, "_args", None) is None:
            self._args = self._get_args()
        return self._args

    def _get_args(self) -> tx.Tuple[tx.Any, ...]:
        return safe_get_args(self.hint, unwrap=self.UNWRAP)

    @property
    def fallback(self) -> tx.Any:
        """A "concrete" fallback type for the type hint, if possible."""
        if getattr(self, "_fallback", None) is None:
            self._fallback = self._get_fallback()
        return self._fallback

    def _get_fallback(self) -> tx.Any:
        try:
            return get_concrete_type(self.hint, self.FALLBACK)
        except TypeError:
            return self.hint

    def __call__(self, *args, **kwargs) -> T:
        """
        Do some magic!

        !!! tip "Subclasses must implement this method."
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement __call__"
        )

    def __repr__(self) -> str:
        # `is not`, not `!=`: a hint with a custom `__eq__` (a numpy-based
        # hint, say) would otherwise make `repr` raise - inside error
        # formatting, of all places. Typing caches its aliases, so
        # identity holds for the hints this compares.
        hint_arg = self.hint if self.hint is not self.DEFAULT else ""
        return f"{type(self).__name__}({hint_arg})"

    def __str__(self) -> str:
        return repr(self)

    def error(
        self, value: tx.Any = UNSET, message: tx.Optional[str] = None,
        **kwargs
    ) -> "MagicError":
        """
        Build a [`MagicError`][] for the given value and message.

        The error is **returned**, not raised, so that the caller keeps
        the `raise` and its traceback starts where the failure is:

        ```python
        raise self.error(value, "Not a valid instance.")
        ```

        !!! tip
            Subclasses override this to build their own error type.
        """
        error_type = kwargs.pop("type", MagicError)
        kwargs.setdefault("this", self)
        kwargs.setdefault("value", value)
        return error_type(message or "", **kwargs)


class MultipleCauses(Exception):
    """A wrapper exception that contains multiple causes."""

    def __init__(self, causes: tx.Iterable[Exception]) -> None:
        super().__init__()
        self.__all_causes__ = tuple(causes)


def _rebuild_magic_error(
    cls: tx.Type["MagicError"],
    message: str,
    args: tx.Tuple[tx.Any, ...],
    this: tx.Any,
    value: tx.Any,
) -> "MagicError":
    """Reconstruct a [`MagicError`][] from its undecorated parts."""
    # Module-level (rather than a lambda or a method) so that it can be
    # pickled by reference.
    return cls(message, *args, this=this, value=value)


class MagicError(Exception):
    """An exception raised by magic objects (factories, converters)."""

    def __new__(cls, *args, **kwargs) -> tx.Self:
        # Avoids errors in python 3.8, where Exception implements its own
        # __new__ but without keyword arguments, making subclasses fail
        # when they are initialized with keyword arguments.
        return super().__new__(cls, *args)

    def __init__(self, *args, **kwargs) -> None:
        """
        Other Parameters
        ----------------
        this : MagicHint
            The MagicHint instance that raised the error.
        value : Any
            The value that caused the error.
        """
        this = kwargs.pop("this", None)
        value = kwargs.pop("value", UNSET)
        self.this = this
        self.value = value
        if args:
            msg, *args = args
        else:
            msg = ""
        self.message = msg
        msg = self._make_message(msg, this=True, value=True, causes=False)
        super().__init__(msg, *args)

    @property
    def nice_message(self) -> str:
        return getattr(self, "args", ("",))[0]

    def __reduce__(self) -> tx.Tuple[tx.Any, ...]:
        # The default `BaseException.__reduce__` returns `(cls, self.args)`,
        # whose first element is the *decorated* message - so a round-trip
        # would decorate it a second time - and it drops `this`/`value`,
        # which live outside `args`. Rebuild from the undecorated parts.
        rest = tuple(self.args[1:])
        state = (type(self), self.message, rest, self.this, self.value)
        return (_rebuild_magic_error, state)

    @property
    def causes(self) -> tx.Tuple[Exception, ...]:
        if hasattr(self, "__all_causes__"):
            return self.__all_causes__
        if self.__cause__ is not None:
            # A `MultipleCauses` wrapper is transparent: expose the causes
            # it carries, not the wrapper itself.
            cause = self.__cause__
            return getattr(cause, "__all_causes__", (cause,))
        return ()

    @property
    def depth(self) -> int:
        return 1 + max(
            (getattr(p, "depth", 0) for p in self.causes), default=0
        )

    @property
    def best_cause(self) -> tx.Optional[tx.Self]:
        return max(
            self.causes,
            key=lambda p: getattr(p, "depth", 0),
            default=None
        )

    def _make_message(
        self,
        message: tx.Optional[str] = None,
        this: bool = True,
        value: bool = True,
        causes: bool = True
    ) -> str:
        if message is None:
            # `self.message` is the undecorated text. Using `nice_message`
            # here (which is `args[0]`, already decorated by `__init__`)
            # would prefix and append a second time at every level.
            message = self.message or ""

        # Only decorate with what was actually supplied. A `MagicError`
        # raised without a `this`/`value` used to render them anyway, as
        # a literal "None: " prefix and a "|> value = <UNSET>" line.
        if this and self.this is not None:
            if message:
                message = f"{self.this!r}: {message}"
            else:
                message = f"{self.this!r}"

        if value and self.value is not UNSET:
            message = f"{message}\n|> value = {self.value!r}"

        if causes and self.causes:
            arrow = "?>" if len(self.causes) > 1 else "->"
            cause_value = len(self.causes) == 1
            for cause in self.causes:
                if hasattr(cause, "_make_message"):
                    cause_message = cause._make_message(
                        this=this, value=cause_value
                    )
                else:
                    cause_message = f"{type(cause).__name__}: {cause}"
                message = f"{message}\n{arrow} {cause_message}"
        return message


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


def get_default(hint: tx.Any) -> tx.Any:
    """
    Get a default value from a type hint.

    * If the hint is a [`Literal`][tx.Literal], the first value in the
      literal is returned ([`None`][], if [`None`][] is one of the
      literal's values).
    * If the hint is a [`Union`][tx.Union] that contains [`NoneType`][],
      [`None`][] is returned.
    * Otherwise, if the hint is a [`Union`][tx.Union], we recurse through
      its sub-hints and return the first default found.
    * If no default value can be found, a [`TypeError`][] is raised.
      A factory should then be used.
    """
    origin = safe_get_origin(hint, unwrap=tx.Annotated)
    args = safe_get_args(hint, unwrap=tx.Annotated)
    if origin is tx.Literal and args:
        if None in args:
            return None
        return args[0]
    if origin in UNION_TYPES:
        if NoneType in args:
            return None
        for arg in args:
            try:
                return get_default(arg)
            except TypeError:
                continue
    raise TypeError(f"Cannot get default for hint {hint}")


def get_from_registry(hint: tx.Any, registry: dict) -> tx.Any:
    """
    Get the best matching value from a registry whose keys are types or
    type hints.

    The best match is the registry key that is the narrowest superclass
    (or superhint) of `hint`, following its MRO; exact matches are always
    preferred. If `hint` is [`Annotated`][typing.Annotated] and no match is
    found for it directly, the search is retried against its unwrapped
    hint.

    !!! example
        ```pycon
        >>> registry = {int: "number", object: "any"}
        >>> get_from_registry(bool, registry)
        'number'
        >>> get_from_registry(str, registry)
        'any'
        ```
    """
    # First naive pass
    best_match, best_dist = _get_best_match(hint, registry)

    # Second pass, where Annotated hints are unwrapped.
    # We only use the resulting match if it is better than the first pass.
    if best_dist != 0 and safe_get_origin(hint) is tx.Annotated:
        hint = safe_get_origin(hint, unwrap=tx.Annotated)
        better_match, better_dist = _get_best_match(hint, registry)
        if better_dist < best_dist:
            best_match, best_dist = better_match, better_dist

    if best_match is not None:
        return registry[best_match]

    return None


def _get_best_match(hint: tx.Any, registry: dict) -> tx.Tuple[tx.Any, float]:
    """
    Get the best matching value from a registry whose keys are types or
    type hints, and return the key and value as a tuple.
    """
    hint = safe_get_origin(hint)

    best_match, best_dist = None, float("inf")
    for key in registry:

        dist = _type_dist(hint, key)

        if dist == float("inf"):
            # Not a match at all.
            continue

        if dist == 0:
            # Perfect match -> stop here
            best_match, best_dist = key, dist
            break

        if best_match is None:
            best_match, best_dist = key, dist
            continue

        key_is_td, best_is_td = is_typeddict(key), is_typeddict(best_match)
        if key_is_td != best_is_td:
            # Prefer a typeddict key over a plain one. Their distances are
            # measured along *different* hierarchies -- `__orig_bases__`
            # for a typeddict, `__mro__` for a class -- so the two numbers
            # are not comparable, and the nearer one is not the better
            # one. A typeddict inheriting from another typeddict sits at
            # distance 2 from `TypedDict` but only 1 from `dict`, and used
            # to be handed to the `dict` entry.
            if key_is_td:
                best_match, best_dist = key, dist
            continue

        if dist < best_dist:
            best_match, best_dist = key, dist

        elif dist == best_dist and safe_issubclass(key, best_match):
            # Prefer more specific subclass
            best_match = key

    return best_match, best_dist


def _type_dist(subcls: type, cls: type) -> float:
    """Distance between two types, based on their inheritance hierarchy."""
    if safe_isinstance(subcls, tx.TypeVar):
        subcls = tx.TypeVar
    if subcls is cls:
        return 0
    if not issubclassable(subcls) or not issubclassable(cls):
        return float("inf")
    if not safe_issubclass(subcls, cls):
        return float("inf")
    # Our `is_typeddict`, not `tx.is_typeddict`: the latter is False for
    # `TypedDict` itself, which would send a typeddict subclass down the
    # `__mro__` branch, where `TypedDict` never appears - so the loop below
    # would fall through and report the "not found" distance instead of 1.
    if is_typeddict(cls):
        cls = _canonical_typeddict(cls)
        bases = _all_orig_bases(subcls)
    else:
        bases = subcls.__mro__
    distance = 0
    for base in bases:
        if base is cls:
            return distance
        distance += 1
    return 1000


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


def ishintstance(obj: tx.Any, hint: tx.Any) -> bool:
    """
    Like isinstance, but the second argument can be a type hint.

    * If `hint` is [`type`][] or [`Type[...]`][tx.Type], checks
      that `obj` is a type and that it is valid subclass of the hint argument.
    * If `hint` is a [`Literal`][tx.Literal], checks that `obj` is one of
      its values. The value must match in type as well: `#!python True` is
      not a valid `#!python Literal[1]`, even though `#!python True == 1`.
    * If `hint` is a [`Union`][tx.Union], checks `obj` against each of
      its members.
    * Otherwise, returns `#!python  issubhint(type(obj), hint)`.

    !!! warning
        A container's **item types are not checked**: a value carries its
        type, and a type carries no arguments, so `#!python [1, 2]` is a
        valid `#!python List[str]` as far as this function is concerned.
        (Python itself refuses `#!python isinstance(x, list[int])` for the
        same reason.) Checking the items means iterating them, which is
        the caller's decision to make - `bagof.validators` does it.

    !!! example
        ```pycon
        >>> ishintstance(1, int)
        True
        >>> ishintstance(1, Literal[1, 2])
        True
        >>> ishintstance(bool, Type[int])
        True
        ```
    """
    hint = normalise_hint(hint)
    if hint is tx.Any:
        return True
    # Resolve typevars too, so a typevar behaves exactly like the hint it
    # stands for - the same rule `MagicHint.UNWRAP` documents.
    hint = unwrap(hint, (tx.Annotated, tx.TypeVar))
    if hint is tx.Any:
        return True
    origin_uw = get_origin_uw(hint)
    if origin_uw is type:
        return _ishintstance_type(obj, hint)
    if origin_uw is tx.Literal:
        return _ishintstance_literal(obj, hint)
    if origin_uw in UNION_TYPES:
        args = get_args_uw(hint)
        if args:
            return any(ishintstance(obj, arg) for arg in args)
    if isinstance(origin_uw, type):
        # Only the origin can be checked here: a value carries its type,
        # and a type carries no arguments - `type([1])` is `list`, never
        # `List[int]`. Checking the arguments means looking at the items,
        # which is the caller's business, not an instance check's.
        return safe_issubclass(type(obj), origin_uw)
    return issubhint(type(obj), hint)


def _ishintstance_literal(obj: tx.Any, hint: tx.Any) -> bool:
    """Check that a value is one of a `Literal`'s values."""
    # Both the type and the value must match. Python compares `True == 1`
    # and `1 == 1.0` as equal, but PEP 586 makes literal matching
    # type-aware, so `Literal[1]` must reject `True` and `1.0`.
    # `eq_safenan` keeps a NaN literal comparable with itself.
    return any(
        type(arg) is type(obj) and eq_safenan(arg) == eq_safenan(obj)
        for arg in get_args_uw(hint)
    )


def _ishintstance_type(obj: tx.Any, hint: tx.Any) -> bool:
    """Like isinstance, but the second argument can be a type hint."""
    # Unwrap the hint, do *not* take its origin: the origin of `type[T]`
    # is the bare `type`, whose `get_args` is always empty, which would
    # make every `type[T]` behave like an unparametrised `type`.
    hint_uw = unwrap(hint)
    if safe_get_origin(hint_uw) is not type:
        # Invalid superhint -> error
        raise TypeError(f"Hint {hint} is not a type[]")
    args_uw = tx.get_args(hint_uw)
    if not args_uw:
        # hint is `type` (or `tx.Type`), so any type is valid
        return isinstance(obj, type)
    # hint is `type[T]` (or `tx.Type[T]`), so check that obj is a subclass of T
    return isinstance(obj, type) and safe_issubclass(obj, args_uw[0])


def issubhint(hint: tx.Any, superhint: tx.Any) -> bool:
    """
    Check that a hint is a sub-hint for another hint.

    A hint is a valid subhint if all values that are valid for the hint
    are also valid for the superhint.

    Arguments are compared covariantly, so `#!python List[bool]` is a
    subhint of `#!python List[int]`. A hint with no arguments is *not* a
    subhint of one that has them - a bare `#!python list` may hold
    anything, so it cannot stand in for a `#!python List[int]`.

    !!! note
        An **unparametrised** `#!python Union` or `#!python Literal` asks
        a different question: *is this hint one of those?* So
        `#!python issubhint(int, Union)` is `#!python False` (an
        `#!python int` is not a union) even though
        `#!python issubhint(int, Union[int, str])` is `#!python True`.
        This makes them usable as a `#!python BOUND`, and it is why the
        relation is not transitive through a bare `#!python Union`.

    !!! example
        ```pycon
        >>> from typing import List, Union
        >>> issubhint(bool, int)
        True
        >>> issubhint(Union[int, str], Union[int, str, bytes])
        True
        >>> issubhint(List[bool], List[int])
        True
        >>> issubhint(list, List[int])  # a bare list may hold anything
        False
        >>> issubhint(int, str)
        False
        ```
    """
    hint, superhint = normalise_hint(hint), normalise_hint(superhint)

    # shortcircuits
    if superhint is tx.Any:
        return True

    if hint is superhint:
        return True

    # Unwrap superhint origin
    origin_uw = get_origin_uw(superhint)

    if origin_uw is tx.Any:
        return True

    if isinstance(origin_uw, tx.TypeVar):
        return _issubtypevar(hint, superhint)

    if isinstance(hint, tx.TypeVar):
        # Unwrap typevar so that its bound can be checked against the
        # superhint. We've already taken care of the case where the
        # superhint is a typevar.

        # For constraints, each constraint must be a subhint of the
        # superhint
        constraints = getattr(hint, "__constraints__", ())
        if constraints:
            if origin_uw in UNION_TYPES:
                # Against a union, ask about the union the typevar
                # stands for: each constraint on its own is not a union,
                # but their combination is.
                return issubhint(unwrap(hint, tx.TypeVar), superhint)
            return all(
                issubhint(constraint, superhint)
                for constraint in constraints
            )

        # For bounds, the bound must be a subhint of the superhint
        return issubhint(unwrap(hint, tx.TypeVar), superhint)

    if origin_uw in UNION_TYPES:
        return _issubunion(hint, superhint)

    if origin_uw is tx.Literal:
        return _issubliteral(hint, superhint)

    if origin_uw is type(None):
        return _issubnone(hint, superhint)

    if origin_uw is type:
        return _issubtype(hint, superhint)

    if isinstance(origin_uw, type):
        return _issubclasshint(hint, superhint, origin_uw)

    return False


def _issubclasshint(hint: tx.Any, superhint: tx.Any, origin: type) -> bool:
    """Check that a hint is a sub-hint for a hint whose origin is a class."""
    # Compare origins, not the hints themselves: a parametrised alias
    # (`List[int]`) and an `Annotated` wrapper are not instances of
    # `type`, so handing either to `safe_issubclass` directly would
    # answer False for every one of them.
    hint_uw = unwrap(hint)
    if not safe_issubclass(get_origin_uw(hint_uw), origin):
        return False

    superargs = safe_get_args(unwrap(superhint))
    if not superargs:
        # An unparametrised superhint constrains nothing further.
        return True

    args = safe_get_args(hint_uw)
    if not args:
        # `list` cannot stand in for `List[int]`: it may hold anything.
        return False

    return _issubargs(args, superargs)


def _issubargs(
    args: tx.Tuple[tx.Any, ...], superargs: tx.Tuple[tx.Any, ...]
) -> bool:
    """Check a hint's arguments against a superhint's, covariantly."""
    # A trailing ellipsis (`Tuple[int, ...]`, `Callable[..., int]`) means
    # "any number of these", so it does not line up positionally.
    if Ellipsis in superargs or Ellipsis in args:
        if Ellipsis not in superargs:
            return False
        if Ellipsis not in args:
            # Every argument must satisfy the repeated one.
            head = tuple(a for a in superargs if a is not Ellipsis)
            return len(head) == 1 and all(
                issubhint(arg, head[0]) for arg in args
            )
        args = tuple(a for a in args if a is not Ellipsis)
        superargs = tuple(a for a in superargs if a is not Ellipsis)

    if len(args) != len(superargs):
        return False

    return all(
        issubhint(arg, superarg) for arg, superarg in zip(args, superargs)
    )


def _issubnone(hint: tx.Any, superhint: tx.Any) -> bool:
    """Check that a hint is a sub-hint for NoneType."""
    none_uw = get_origin_uw(superhint)
    if none_uw is not type(None):
        raise TypeError(f"nonehint {superhint} is not a NoneType")
    origin_uw = get_origin_uw(hint)
    return origin_uw is type(None)


def _issubliteral(hint: tx.Any, superhint: tx.Any) -> bool:
    """Check that a hint is a sub-hint for a Literal."""
    hint_uw = unwrap(hint)
    superhint_uw = unwrap(superhint)
    if safe_get_origin(superhint_uw) is not tx.Literal:
        # Superhint is not a literal -> error
        raise TypeError(f"Super-hint {superhint} is not a Literal")
    if safe_get_origin(hint_uw) is not tx.Literal:
        # Hint is not a Literal, cannot be a subhint
        return False
    # !! We use tx.get_origin instead of _get_origin
    # !! to differentiate tx.Literal (origin is None)
    # !! from tx.Literal[()] (origin is tx.Literal)
    if not tx.get_origin(superhint_uw):
        # All literals are subhints of `tx.Literal`
        return True
    if not tx.get_origin(hint_uw):
        # # tx.Literal is not a subhint of tx.Literal[...]
        return False
    # Check that all args of hint are in superhint
    args = safe_get_args(hint_uw)
    superargs = safe_get_args(superhint_uw)
    return all(arg in superargs for arg in args)


def _issubtypevar(hint: tx.Any, superhint: tx.TypeVar) -> bool:
    """Check that a hint is a sub-hint for a TypeVar."""
    hint_uw = unwrap(hint)
    superhint_uw = unwrap(superhint)
    if not isinstance(superhint_uw, tx.TypeVar):
        # Invalid superhint -> error
        raise TypeError(f"Super-hint {superhint} is not a TypeVar")
    if hint_uw is superhint_uw:
        # Exact match
        return True
    if getattr(superhint_uw, "__constraints__", ()):
        # If constraints, check that hint is a subhint of one of them
        constraints = superhint_uw.__constraints__
        for constraint in constraints:
            if issubhint(hint, constraint):
                return True
        # Else, if hint is a TypeVar, check that all its constraints are
        # subhints of one of the superhint's constraints
        if isinstance(hint_uw, tx.TypeVar):
            subconstraints = getattr(hint_uw, "__constraints__", ())
            if not subconstraints:
                return False
            return all(
                any(
                    issubhint(subconstraint, constraint)
                    for constraint in constraints
                )
                for subconstraint in subconstraints
            )
        # Otherwise, constraints do not match
        return False
    elif getattr(superhint_uw, "__bound__", None) is not None:
        # If bound, check that hint is a subhint of the bound
        bound = superhint_uw.__bound__
        if issubhint(hint, bound):
            return True
        # Else, if hint is a TypeVar, check that all its constraints are
        # subhints of one of the superhint's constraints
        if isinstance(hint_uw, tx.TypeVar):
            # If hint is a TypeVar, check that all its bound is a
            # subhint of the superhint's bound
            subbound = getattr(hint_uw, "__bound__", None)
            if subbound is None:
                return False
            return issubhint(subbound, bound)
        # Otherwise, bound does not match
        return False
    else:
        # Unconstrained TypeVar -> any hint is a subhint
        return True


def _issubunion(hint: tx.Any, superhint: tx.Any) -> bool:
    """Check that a hint is a sub-hint for a Union."""
    hint_uw = unwrap(hint)
    superhint_uw = unwrap(superhint)
    if safe_get_origin(superhint_uw) not in UNION_TYPES:
        # Invalid superhint -> error
        raise TypeError(f"union {superhint} is not a Union type")
    # !! We use tx.get_origin instead of _get_origin
    # !! to differentiate tx.Union (origin is None)
    # !! from tx.Union[...] (origin is tx.Union)
    if not tx.get_origin(superhint_uw):
        # Every union - and only a union - is a subhint of the bare
        # `tx.Union`, the same rule `_issubliteral` applies for a bare
        # `Literal`. A *parametrised* union is different: a plain type
        # is a subhint of one that contains it, which the member logic
        # below works out.
        return safe_get_origin(hint_uw) in UNION_TYPES
    # Collect the hint's member hints. A hint is a subhint of the union if
    # each of its members is a subhint of one of the union's members:
    #   * a parametrised union contributes its arguments;
    #   * the bare `tx.Union` is not a subhint of a parametrised union;
    #   * any other hint (e.g. `int`) is a single member, so that a plain
    #     type is a subhint of a union that contains it.
    if safe_get_origin(hint_uw) in UNION_TYPES:
        if not tx.get_origin(hint_uw):
            return False
        members = safe_get_args(hint_uw)
    else:
        members = (hint_uw,)
    superargs = safe_get_args(superhint_uw)
    return all(
        any(issubhint(member, superarg) for superarg in superargs)
        for member in members
    )


def _issubtype(hint: tx.Any, superhint: tx.Any) -> bool:
    """Check that a hint is a sub-hint for a type[...] hint."""
    hint_uw = unwrap(hint)
    superhint_uw = unwrap(superhint)
    if safe_get_origin(superhint_uw) is not type:
        # Invalid superhint -> error
        raise TypeError(f"superhint {superhint} is not a type")
    if safe_get_origin(hint_uw) is not type:
        # Hint is not a type, cannot be a subhint
        return False
    if not tx.get_args(superhint_uw):
        # All types are subhints of `tx.Type`
        return True
    if not tx.get_args(hint_uw):
        # # tx.Type is not a subhint of tx.Type[...]
        return False
    # Check that the hint's arg is a subclass of the superhint's arg
    args = safe_get_args(hint_uw)
    superargs = safe_get_args(superhint_uw)
    return safe_issubclass(args[0], superargs[0])


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
    (abc.Callable, "Callable"),
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
