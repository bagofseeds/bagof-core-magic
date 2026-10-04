"""
Compatibility shims and constants shared by the whole package.

Spellings that differ across the supported Python versions (`NoneType`,
`UnionType`, the typing constructs that are classes on some versions
only), and the [`UNSET`][bagof.core.magic.UNSET] sentinel.

The public names are re-exported by [`bagof.core.magic`][]; import them
from there.
"""

__all__ = [
    "NoneType",
    "UnionType",
    "UNION_TYPES",
    "REAL_TYPES",
    "Unset",
    "UNSET",
]

# stdlib
import numbers

# dependencies
import typing_extensions as tx

# optionals
if tx.TYPE_CHECKING:
    from types import NoneType, UnionType
else:
    try:
        from types import NoneType, UnionType
    except ImportError:  # pragma: no cover  -- Python < 3.10
        NoneType = type(None)
        UnionType = tx.Union


# constants
UNION_TYPES = (
    (tx.Union,) if UnionType is tx.Union else (tx.Union, UnionType)
)
"""The union spellings this package understands."""

REAL_TYPES = (numbers.Real,)
"""
The real-number types [`eq_safenan`][] recognises.

!!! note
    NumPy registers `numpy.floating` as a [`numbers.Real`][], so NumPy
    floats are recognised without this package importing NumPy.
"""

_SPECIAL_FORMS = (tx.Any, tx.Optional, tx.Literal, tx.Annotated) + UNION_TYPES
"""
The typing constructs that must never be treated as classes.

Several of these *are* classes on some Python versions and not on others
- [`Any`][typing.Any] became one in 3.11, [`Annotated`][typing.Annotated]
was one through 3.12 but not from 3.13, and [`Union`][typing.Union] became
one in 3.14, when it merged with [`types.UnionType`][] - so
`#!python isinstance(hint, type)` silently gives different answers across
the versions this package supports. Pin the answer instead of inheriting
it.
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
