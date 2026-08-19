"""Tests for the hint-introspection helpers."""

# dependencies
import pytest
import typing_extensions as tx

# locals
from bagof.core.magic import (
    _type_dist,
    get_from_registry,
    ishintstance,
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
