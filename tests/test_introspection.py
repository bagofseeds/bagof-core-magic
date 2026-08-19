"""Tests for the hint-introspection helpers."""

# dependencies
import pytest
import typing_extensions as tx

# locals
from bagof.core.magic import (
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
