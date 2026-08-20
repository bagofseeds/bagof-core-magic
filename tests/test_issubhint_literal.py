"""Tests for `issubhint` with a `Literal` super-hint."""

# dependencies
import pytest
import typing_extensions as tx

# locals
from bagof.core.magic import _issubliteral, issubhint

L = tx.Literal

# (hint, superhint, expected)
LITERAL_CASES = [
    # A literal is a subhint of one that contains all of its values.
    (L[1], L[1, 2], True),
    (L[1, 2], L[1, 2, 3], True),
    (L[1, 2], L[1], False),
    (L["a"], L["a", "b"], True),
    (L["a"], L["b"], False),
    (L[None], L[None, 1], True),
    # A mixed literal needs every value present.
    (L[1, "a"], L[1, "a", 2], True),
    (L[1, "a"], L[1, 2], False),
    # Every literal is a subhint of the bare `Literal`...
    (L[1], L, True),
    (L, L, True),
    # ... but the bare `Literal` is not a subhint of a parametrised one.
    (L, L[1, 2], False),
    # A non-literal hint is never a subhint of a literal.
    (int, L[1, 2], False),
    (tx.Union[int, str], L[1, 2], False),
    # `Annotated` is transparent on both sides.
    (tx.Annotated[L[1], "meta"], L[1, 2], True),
    (L[1], tx.Annotated[L[1, 2], "meta"], True),
]


@pytest.mark.parametrize(
    "hint,superhint,expected",
    LITERAL_CASES,
    ids=[f"{h}<:{s}" for h, s, _ in LITERAL_CASES],
)
def test_issubhint_literal(
    hint: tx.Any, superhint: tx.Any, expected: bool
) -> None:
    assert issubhint(hint, superhint) is expected


def test_issubliteral_rejects_a_non_literal_superhint() -> None:
    with pytest.raises(TypeError, match="is not a Literal"):
        _issubliteral(L[1], int)
