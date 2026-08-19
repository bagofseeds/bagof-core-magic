"""Tests for the hint-introspection helpers."""

# dependencies
import pytest
import typing_extensions as tx

# locals
from bagof.core.magic import (
    _type_dist,
    _unwrap_typevar,
    get_default,
    get_from_registry,
    ishintstance,
    unwrap,
)


class Base(tx.TypedDict):
    a: int


class Middle(Base):
    b: int


class Leaf(Middle):
    c: int


# --- ishintstance against `type[T]` -----------------------------------


@pytest.mark.parametrize(
    "obj,hint,expected",
    [
        # Regression: the argument of `type[T]` used to be ignored, so
        # every class validated against every `type[...]`.
        (bool, tx.Type[int], True),
        (str, tx.Type[int], False),
        (int, tx.Type[int], True),
        # A bare `type` still accepts any class...
        (str, tx.Type, True),
        # ... and neither form accepts a non-class.
        (3, tx.Type[int], False),
        (3, tx.Type, False),
        # `Annotated` is transparent here, like everywhere else.
        (bool, tx.Annotated[tx.Type[int], "meta"], True),
        (str, tx.Annotated[tx.Type[int], "meta"], False),
    ],
)
def test_ishintstance_type(obj: tx.Any, hint: tx.Any, expected: bool) -> None:
    assert ishintstance(obj, hint) is expected


# --- registry resolution ----------------------------------------------


def test_registry_prefers_the_nearest_typeddict() -> None:
    # Regression: a typeddict key at a *worse* distance used to displace a
    # nearer one, so the answer depended on registry insertion order.
    assert get_from_registry(Leaf, {Middle: "near", Base: "far"}) == "near"


def test_registry_result_is_insertion_order_independent() -> None:
    forwards = get_from_registry(Leaf, {Middle: "near", Base: "far"})
    backwards = get_from_registry(Leaf, {Base: "far", Middle: "near"})
    assert forwards == backwards == "near"


def test_registry_still_prefers_an_exact_match() -> None:
    assert get_from_registry(Leaf, {Leaf: "exact", Base: "far"}) == "exact"


def test_registry_documented_example_is_unchanged() -> None:
    registry = {int: "number", object: "any"}
    assert get_from_registry(bool, registry) == "number"
    assert get_from_registry(str, registry) == "any"


def test_typeddict_is_one_step_from_TypedDict() -> None:
    # Regression: `_type_dist` used `tx.is_typeddict`, which is False for
    # `TypedDict` itself, so a typeddict subclass was measured down its
    # `__mro__` - where `TypedDict` never appears - and reported the
    # "not found" distance instead of 1.
    assert _type_dist(Base, tx.TypedDict) == 1
    assert _type_dist(Middle, tx.TypedDict) == 2


# --- get_default -------------------------------------------------------


def test_get_default_of_a_bare_literal_raises_type_error() -> None:
    # Regression: `args[0]` was unguarded, so this raised IndexError -
    # which callers built on the documented TypeError do not catch.
    with pytest.raises(TypeError):
        get_default(tx.Literal)


@pytest.mark.parametrize(
    "hint,expected",
    [
        (tx.Literal[1, 2], 1),
        (tx.Literal[None, 1], None),
        (tx.Optional[int], None),
        (tx.Annotated[tx.Optional[int], "meta"], None),
        (tx.Union[tx.Literal[3], str], 3),
    ],
)
def test_get_default_is_unchanged(hint: tx.Any, expected: tx.Any) -> None:
    assert get_default(hint) == expected


# --- typevar cycles ----------------------------------------------------


def test_the_reentrancy_guard_terminates() -> None:
    # Regression: the guard returned the typevar, which sent `unwrap`
    # straight back into it - turning one infinite recursion into
    # another. Drive the guard directly, so this holds on every Python.
    typevar = tx.TypeVar("typevar")
    assert _unwrap_typevar(typevar, (typevar,)) is tx.Any


def test_unwrap_terminates_on_a_typevar_cycle() -> None:
    first = tx.TypeVar("first")
    second = tx.TypeVar("second")
    try:
        first.__default__ = second
        second.__default__ = first
    except AttributeError:  # pragma: no cover
        # `__default__` is a read-only slot from python 3.13 on, so the
        # cycle cannot be built there. The guard itself is covered above.
        pytest.skip("TypeVar.__default__ is not writable")
    assert unwrap(first, (tx.Annotated, tx.TypeVar)) is tx.Any
