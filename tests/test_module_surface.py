"""Tests for the module's public surface, constants and small helpers."""

# stdlib
import collections
from collections import abc

# dependencies
import pytest
import typing_extensions as tx

# locals
import bagof.core.magic as magic
from bagof.core.magic import (
    UNION_TYPES,
    type2hint,
)

# --- the public surface ------------------------------------------------


def test_every_exported_name_resolves() -> None:
    missing = [name for name in magic.__all__ if not hasattr(magic, name)]
    assert missing == []


@pytest.mark.parametrize(
    "name",
    ["NoneType", "REAL_TYPES", "UnionType", "Unset", "type2hint"],
)
def test_public_names_are_exported(name: str) -> None:
    # `bagof-magic` imports `UnionType` from here, and `type2hint` is a
    # documented public function; both were absent from `__all__`, so the
    # API reference never rendered them.
    assert name in magic.__all__


def test_numpy_is_not_part_of_the_public_surface() -> None:
    assert not hasattr(magic, "np")


def test_union_types_has_no_duplicate() -> None:
    # On Python < 3.10 the `UnionType` fallback is `tx.Union` itself, so
    # the constant used to hold the same object twice.
    assert len(set(UNION_TYPES)) == len(UNION_TYPES)
    assert tx.Union in UNION_TYPES


def test_type_checking_shim_names_come_from_types() -> None:
    # Regression: the `TYPE_CHECKING` branch imported these from
    # `typing_extensions`, which exports neither - and that branch is the
    # only one a type checker ever sees.
    assert not hasattr(tx, "NoneType")
    assert not hasattr(tx, "UnionType")
    assert magic.NoneType is type(None)


# --- type2hint ---------------------------------------------------------


@pytest.mark.parametrize("value", [3, None, [1, 2], "text"])
def test_type2hint_returns_a_non_type_as_is(value: tx.Any) -> None:
    # Regression: `x.__name__` was unguarded, so a non-type raised
    # AttributeError despite the documented "returned as is".
    assert type2hint(value) == value


@pytest.mark.parametrize(
    "cls,name",
    [
        (frozenset, "FrozenSet"),
        (collections.defaultdict, "DefaultDict"),
        (collections.OrderedDict, "OrderedDict"),
        (abc.Set, "AbstractSet"),
    ],
)
def test_type2hint_maps_camel_cased_aliases(cls: type, name: str) -> None:
    # `str.capitalize()` lower-cases everything after the first character,
    # so these never resolved. Subscriptable types are returned unchanged
    # (they need no alias), which is what happens from Python 3.9 on.
    expected = cls if magic.issubscriptable(cls) else getattr(tx, name)
    assert type2hint(cls) is expected
