"""Tests for `MagicError`'s cause chain, message and pickling."""

# stdlib
import pickle

# dependencies
import pytest
import typing_extensions as tx

# locals
from bagof.core.magic import UNSET, MagicError, MagicHint, MultipleCauses


class Magic(MagicHint):
    """A bare magic object, to attach errors to."""

    DEFAULT = tx.Any


def chain(*messages: str) -> MagicError:
    """Build a chain of errors, outermost first."""
    errors = [MagicError(msg, value=i) for i, msg in enumerate(messages)]
    for outer, inner in zip(errors, errors[1:]):
        outer.__cause__ = inner
    return errors[0]


def test_depth_of_a_leaf_error() -> None:
    # Regression: `max()` over an empty `causes` used to raise ValueError,
    # which made `depth` unusable on every error.
    assert MagicError("boom").depth == 1


def test_depth_counts_the_whole_chain() -> None:
    assert chain("top", "mid", "leaf").depth == 3


def test_best_cause_of_a_leaf_error_is_none() -> None:
    assert MagicError("boom").best_cause is None


def test_best_cause_picks_the_deepest_branch() -> None:
    deep = chain("deep", "deeper", "deepest")
    shallow = MagicError("shallow")
    top = MagicError("top")
    top.__all_causes__ = (shallow, deep)
    assert top.best_cause is deep


def test_message_is_not_decorated_twice() -> None:
    # Regression: `_make_message` fell back to `nice_message` (already
    # decorated by `__init__`), so every level repeated its prefix and
    # its value line.
    error = MagicError("failed", value=1, this=Magic())
    message = error._make_message()
    assert message.count("Magic()") == 1
    assert message.count("|> value =") == 1
    assert message == "Magic(): failed\n|> value = 1"


def test_nested_message_reports_each_level_once() -> None:
    message = chain("top", "leaf")._make_message()
    assert message.count("|> value =") == 2
    assert message.splitlines() == [
        "None: top",
        "|> value = 0",
        "-> None: leaf",
        "|> value = 1",
    ]


def test_multiple_causes_is_transparent() -> None:
    # Regression: `causes` used to return the `MultipleCauses` wrapper
    # itself, so the errors it carries were invisible to `depth`,
    # `best_cause` and `_make_message` alike.
    first = MagicError("first")
    second = MagicError("second")
    error = MagicError("none matched")
    error.__cause__ = MultipleCauses([first, second])

    assert error.causes == (first, second)
    assert error.depth == 2
    assert error.best_cause in (first, second)


def test_non_magic_causes_are_rendered() -> None:
    # `MultipleCauses` usually carries plain `TypeError`/`ValueError`s.
    error = MagicError("none matched")
    error.__cause__ = MultipleCauses([ValueError("bad value"), TypeError()])
    message = error._make_message()
    assert "ValueError: bad value" in message
    assert "TypeError" in message


def test_single_cause_arrow_has_no_double_space() -> None:
    assert "->  " not in chain("top", "leaf")._make_message()


def test_multiple_cause_arrow_has_no_double_space() -> None:
    error = MagicError("top")
    error.__all_causes__ = (MagicError("a"), MagicError("b"))
    assert "?>  " not in error._make_message()


@pytest.mark.parametrize("value", [3, None, "text", [1, 2]])
def test_pickle_round_trip_preserves_everything(value: tx.Any) -> None:
    # Regression: the default `BaseException.__reduce__` re-decorated the
    # message and dropped `this`/`value`.
    error = MagicError("boom", value=value)
    revived = pickle.loads(pickle.dumps(error))
    assert type(revived) is MagicError
    assert revived.args == error.args
    assert revived.message == "boom"
    assert revived.value == value
    assert revived.this is None


def test_pickle_round_trip_of_a_subclass() -> None:
    class Custom(MagicError):
        pass

    # A subclass defined at module scope is required for pickling; use the
    # exception type the downstream packages actually derive.
    error = MagicError("boom", value=1)
    assert pickle.loads(pickle.dumps(error)).value == 1
    assert issubclass(Custom, MagicError)


# --- make_error builds, error raises ----------------------------------


def test_make_error_returns_and_error_raises() -> None:
    # The two verbs are distinct, so neither can be mistaken for the
    # other - `MagicHint.error` used to raise while every subclass
    # override returned, with the arguments in the opposite order.
    magic = Magic()
    built = magic.make_error(1, "built")
    assert isinstance(built, MagicError)
    assert built.value == 1
    assert built.message == "built"

    with pytest.raises(MagicError) as info:
        magic.error("raised", 1)
    assert info.value.value == 1
    assert info.value.message == "raised"


def test_error_raises_whatever_make_error_builds() -> None:
    class CustomError(MagicError):
        pass

    class CustomMagic(MagicHint):
        DEFAULT = tx.Any

        def make_error(
            self,
            value: tx.Any = UNSET,
            message: tx.Optional[str] = None,
            **kwargs,
        ) -> MagicError:
            kwargs.setdefault("type", CustomError)
            return super().make_error(value, message, **kwargs)

    # Overriding only `make_error` is enough: `error` is inherited.
    assert isinstance(CustomMagic().make_error(1, "x"), CustomError)
    with pytest.raises(CustomError):
        CustomMagic().error("x", 1)


# --- a hint cannot be reassigned --------------------------------------


def test_hint_cannot_be_reassigned() -> None:
    # The introspected properties are computed once and `__post_init__`
    # runs once, so a reassigned hint would leave the object describing
    # a hint it no longer has - and skip its own BOUND check.
    magic = Magic(tx.List[int])
    assert magic.origin is list
    with pytest.raises(AttributeError, match="cannot be reassigned"):
        magic.hint = tx.Dict[str, int]
    assert magic.hint == tx.List[int]
    assert magic.origin is list


def test_other_attributes_are_still_writable() -> None:
    magic = Magic(int)
    magic.whatever = 1  # type: ignore[attr-defined]
    assert magic.whatever == 1  # type: ignore[attr-defined]
