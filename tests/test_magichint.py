"""Tests for `MagicHint`'s introspection of a type hint."""

# stdlib
from collections import abc

# dependencies
import pytest
import typing_extensions as tx

# locals
from bagof.core.magic import MagicHint


class Magic(MagicHint):
    """A bare magic object, to introspect hints with."""

    DEFAULT = tx.Any


class KeepsTypeVars(MagicHint):
    """A magic object that opts out of resolving typevars."""

    DEFAULT = tx.Any
    UNWRAP = (tx.Annotated,)


BOUND_TO_LIST = tx.TypeVar("BOUND_TO_LIST", bound=tx.List[int])
BOUND_TO_INT = tx.TypeVar("BOUND_TO_INT", bound=int)
CONSTRAINED = tx.TypeVar("CONSTRAINED", int, str)
UNBOUND = tx.TypeVar("UNBOUND")
WITH_DEFAULT = tx.TypeVar("WITH_DEFAULT", bound=tx.Sequence[int],
                          default=tx.List[int])


@pytest.mark.parametrize(
    "hint,expected",
    [
        # Not a typevar: unchanged.
        (tx.List[int], tx.List[int]),
        (int, int),
        (tx.Any, tx.Any),
        # `Annotated` is unwrapped.
        (tx.Annotated[tx.List[int], "meta"], tx.List[int]),
        # A typevar resolves to its bound...
        (BOUND_TO_LIST, tx.List[int]),
        (BOUND_TO_INT, int),
        # ... or to the union of its constraints...
        (CONSTRAINED, tx.Union[int, str]),
        # ... or to `Any` when it has neither.
        (UNBOUND, tx.Any),
        # A default takes precedence over the bound.
        (WITH_DEFAULT, tx.List[int]),
        # `Annotated` and typevars unwrap together.
        (tx.Annotated[BOUND_TO_LIST, "meta"], tx.List[int]),
    ],
)
def test_unwrapped(hint: tx.Any, expected: tx.Any) -> None:
    assert Magic(hint).unwrapped == expected


@pytest.mark.parametrize(
    "typevar,equivalent",
    [
        (BOUND_TO_LIST, tx.List[int]),
        (BOUND_TO_INT, int),
        (CONSTRAINED, tx.Union[int, str]),
        (UNBOUND, tx.Any),
        (WITH_DEFAULT, tx.List[int]),
        (tx.Annotated[BOUND_TO_LIST, "meta"], tx.List[int]),
    ],
)
def test_typevar_introspects_like_the_hint_it_stands_for(
    typevar: tx.Any, equivalent: tx.Any
) -> None:
    # Every introspection property must agree, so that a consumer walking a
    # hint's structure cannot see a typevar as an argument-less hint.
    magic, expected = Magic(typevar), Magic(equivalent)
    assert magic.unwrapped == expected.unwrapped
    assert magic.origin == expected.origin
    assert magic.args == expected.args


def test_args_are_not_silently_empty_for_a_typevar() -> None:
    # Regression: `args` used to be empty for a typevar, so consumers
    # iterating over a container's arguments silently did nothing.
    assert Magic(BOUND_TO_LIST).args == (int,)
    assert Magic(BOUND_TO_LIST).origin is list


def test_fallback_agrees_with_origin_for_a_typevar() -> None:
    # `fallback` has always resolved typevars (via `get_concrete_type`);
    # the other properties must not contradict it.
    magic = Magic(BOUND_TO_LIST)
    assert magic.fallback is list
    assert magic.origin is list


@pytest.mark.parametrize(
    "hint,origin,args",
    [
        (tx.List[int], list, (int,)),
        (tx.Mapping[str, int], abc.Mapping, (str, int)),
        (int, int, ()),
        (tx.Annotated[tx.List[int], "meta"], list, (int,)),
    ],
)
def test_non_typevar_hints_are_unaffected(
    hint: tx.Any, origin: tx.Any, args: tx.Tuple[tx.Any, ...]
) -> None:
    assert Magic(hint).origin is origin
    assert Magic(hint).args == args


def test_unwrap_is_overridable() -> None:
    # Opting out leaves typevars unresolved.
    assert KeepsTypeVars(BOUND_TO_LIST).unwrapped is BOUND_TO_LIST
    assert KeepsTypeVars(BOUND_TO_LIST).origin is BOUND_TO_LIST
    assert KeepsTypeVars(BOUND_TO_LIST).args == ()
    # `Annotated` is still unwrapped.
    hint = tx.Annotated[tx.List[int], "meta"]
    assert KeepsTypeVars(hint).unwrapped == tx.List[int]


def test_properties_are_cached() -> None:
    magic = Magic(BOUND_TO_LIST)
    assert magic.unwrapped is magic.unwrapped
    assert magic.origin is magic.origin
    assert magic.args is magic.args
