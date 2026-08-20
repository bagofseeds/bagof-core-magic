"""`has_explicit_hint` and `rebind`."""

# dependencies
import pytest
import typing_extensions as tx

# bags
from bagof.core.magic import UNSET, MagicHint


class Configured(MagicHint):
    """A magic object carrying configuration beyond its hint."""

    DEFAULT = int

    def __init__(self, threshold: int, hint: tx.Any = UNSET) -> None:
        super().__init__(hint)
        self.threshold = threshold


def test_has_explicit_hint_is_false_when_defaulted() -> None:
    assert MagicHint().has_explicit_hint is False
    assert Configured(5).has_explicit_hint is False


def test_has_explicit_hint_is_true_when_given() -> None:
    assert MagicHint(int).has_explicit_hint is True
    assert Configured(5, str).has_explicit_hint is True


def test_a_hint_equal_to_the_default_is_still_explicit() -> None:
    # The distinction the `Annotated` handlers rely on: passing the
    # default is not the same as passing nothing.
    assert Configured(5, Configured.DEFAULT).has_explicit_hint is True


def test_rebind_returns_a_copy_with_the_new_hint() -> None:
    original = Configured(5)
    rebound = original.rebind(str)
    assert rebound is not original
    assert rebound.hint is str
    assert original.hint is int


def test_rebind_carries_configuration_over() -> None:
    assert Configured(5).rebind(str).threshold == 5


def test_rebind_marks_the_hint_explicit() -> None:
    assert Configured(5).rebind(str).has_explicit_hint is True


def test_rebind_recomputes_the_memoised_properties() -> None:
    hint = MagicHint(tx.List[int])
    # Materialise the caches first, so a stale one would show.
    assert hint.origin is list
    assert hint.args == (int,)

    rebound = hint.rebind(tx.Dict[str, int])
    assert rebound.origin is dict
    assert rebound.args == (str, int)
    # The original keeps its own answers.
    assert hint.origin is list
    assert hint.args == (int,)


def test_rebind_normalises_the_hint() -> None:
    assert MagicHint(int).rebind(None).hint is type(None)


def test_rebind_runs_post_init() -> None:

    class Bounded(MagicHint):
        BOUND = int

    with pytest.raises(TypeError):
        Bounded(int).rebind(str)


def test_hint_is_still_frozen_on_the_original() -> None:
    hint = MagicHint(int)
    with pytest.raises(AttributeError):
        hint.hint = str


def test_rebind_leaves_the_original_frozen() -> None:
    rebound = MagicHint(int).rebind(str)
    with pytest.raises(AttributeError):
        rebound.hint = float
