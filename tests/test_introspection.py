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
    safe_isinstance,
    safe_issubclass,
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


def test_registry_prefers_TypedDict_over_dict_at_equal_distance() -> None:
    # Both keys are one step away; the typeddict entry is the specific one.
    registry = {dict: "dict", tx.TypedDict: "typeddict"}
    assert get_from_registry(Base, registry) == "typeddict"


def test_registry_typeddict_preference_is_order_independent() -> None:
    # Regression: `safe_issubclass(dict, SomeTypedDict)` used to be True,
    # so the equal-distance tie-break let a `dict` key displace the
    # typeddict one whenever it came second in the registry.
    registry = {tx.TypedDict: "typeddict", dict: "dict"}
    assert get_from_registry(Base, registry) == "typeddict"


def test_registry_ignores_an_unrelated_typeddict_key() -> None:
    # Regression: an unrelated typeddict key (distance `inf`) must not win
    # by tying with the initial infinite `best_dist`.
    assert get_from_registry(int, {tx.TypedDict: "typeddict"}) is None
    registry = {tx.TypedDict: "typeddict", int: "number"}
    assert get_from_registry(int, registry) == "number"


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


# --- tuples, like the builtins ----------------------------------------


@pytest.mark.parametrize(
    "obj,classes,expected",
    [
        (1, (int, str), True),
        (1.5, (int, str), False),
        (1, (), False),
        # Still safe: a non-type member is skipped, not raised on.
        (1, (int, "not a type"), True),
        (1.5, (int, "not a type"), False),
        # A TypedDict cannot be instance-checked at all, so a dict is
        # not an instance of one - see `test_typeddict_is_not_instance
        # _checkable` below.
        ({"a": 1}, (Base,), False),
    ],
)
def test_safe_isinstance_accepts_a_tuple(
    obj: tx.Any, classes: tx.Any, expected: bool
) -> None:
    assert safe_isinstance(obj, classes) is expected


@pytest.mark.parametrize(
    "subcls,classes,expected",
    [
        (bool, (int, str), True),
        (float, (int, str), False),
        (bool, (), False),
        (bool, (int, "not a type"), True),
    ],
)
def test_safe_issubclass_accepts_a_tuple(
    subcls: tx.Any, classes: tx.Any, expected: bool
) -> None:
    assert safe_issubclass(subcls, classes) is expected


# --- TypedDicts are not instance-checkable ----------------------------


def test_typeddict_is_not_instance_checkable() -> None:
    # Python refuses `isinstance(value, SomeTypedDict)` outright, and a
    # TypedDict leaves no trace on the dict it describes - so there is
    # nothing to recognise at runtime.
    with pytest.raises(TypeError):
        isinstance({"a": 1}, Base)
    assert safe_isinstance({"a": 1}, Base) is False
    assert safe_isinstance({"wrong": 1}, Base) is False


def test_dict_is_not_a_subclass_of_a_typeddict() -> None:
    # Regression: an `or subcls is dict` clause ran the relation
    # backwards. A TypedDict is a dict; a dict is not a TypedDict.
    assert safe_issubclass(dict, Base) is False
    assert safe_issubclass(Base, tx.TypedDict) is True
    assert safe_issubclass(Middle, Base) is True


def test_a_bare_dict_hint_does_not_resolve_to_a_typeddict_entry() -> None:
    assert get_from_registry(dict, {Base: "typeddict"}) is None
