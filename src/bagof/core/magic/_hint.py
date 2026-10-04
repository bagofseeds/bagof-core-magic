"""
[`MagicHint`][bagof.core.magic.MagicHint], the base class of magic objects.

The public names are re-exported by [`bagof.core.magic`][]; import them
from there.
"""

__all__ = [
    "MagicHint",
]

# stdlib
import copy

# dependencies
import typing_extensions as tx

# locals
from ._compat import UNSET
from ._errors import MagicError
from ._introspect import (
    get_concrete_type,
    normalise_hint,
    safe_get_args,
    safe_get_origin,
    unwrap,
)
from ._relation import issubhint

# typing
T = tx.TypeVar("T", covariant=True)


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
