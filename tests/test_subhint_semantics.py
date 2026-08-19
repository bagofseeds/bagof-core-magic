"""Tests for `issubhint`/`ishintstance`'s hint semantics."""

# dependencies
import typing_extensions as tx

# locals
from bagof.core.magic import (
    get_concrete_type,
)

ARG_CASES = [
    # A parametrised hint is a subhint of its own origin.
    (tx.List[int], list, True),
    (tx.Dict[str, int], dict, True),
    (tx.List[int], object, True),
    # `Annotated` is transparent, in both directions.
    (tx.Annotated[int, "meta"], int, True),
    (int, tx.Annotated[int, "meta"], True),
    (tx.Annotated[tx.List[int], "meta"], list, True),
    # A bare origin cannot stand in for a parametrised hint: it may hold
    # anything at all.
    (list, tx.List[int], False),
    (dict, tx.Dict[str, int], False),
    # Arguments are compared covariantly.
    (tx.List[bool], tx.List[int], True),
    (tx.List[str], tx.List[int], False),
    (tx.Dict[str, bool], tx.Dict[str, int], True),
    (tx.List[int], tx.List[tx.Any], True),
    # ... including through the container hierarchy.
    (tx.List[int], tx.Sequence[int], True),
    (tx.List[int], tx.Iterable[int], True),
    (tx.List[bool], tx.Sequence[int], True),
    (tx.List[str], tx.Sequence[int], False),
    # Arity must match.
    (tx.Tuple[int], tx.Tuple[int, str], False),
    (tx.Tuple[int, str], tx.Tuple[int, str], True),
    # A trailing ellipsis means "any number of these".
    (tx.Tuple[bool, bool], tx.Tuple[int, ...], True),
    (tx.Tuple[str, int], tx.Tuple[int, ...], False),
    (tx.Tuple[int, ...], tx.Tuple[int, ...], True),
]


# --- get_concrete_type takes the first constraint ---------------------


def test_first_constraint_is_used() -> None:
    assert get_concrete_type(tx.TypeVar("c", int, str)) is int
    # ... in preference to the fallback.
    assert get_concrete_type(tx.TypeVar("c", int, str), list) is int


def test_a_default_still_wins_over_the_constraints() -> None:
    with_default = tx.TypeVar("d", int, str, default=str)
    assert get_concrete_type(with_default) is str


def test_an_abstract_constraint_is_skipped() -> None:
    partly_abstract = tx.TypeVar("a", tx.Sequence[int], int)
    assert get_concrete_type(partly_abstract) is int
