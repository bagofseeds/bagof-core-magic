"""
The errors raised by magic objects.

The public names are re-exported by [`bagof.core.magic`][]; import them
from there.
"""

__all__ = [
    "MultipleCauses",
    "MagicError",
]

# dependencies
import typing_extensions as tx

# locals
from ._compat import UNSET


class MultipleCauses(Exception):
    """A wrapper exception that contains multiple causes."""

    def __init__(self, causes: tx.Iterable[Exception]) -> None:
        super().__init__()
        self.__all_causes__ = tuple(causes)


def _rebuild_magic_error(
    cls: tx.Type["MagicError"],
    message: str,
    args: tx.Tuple[tx.Any, ...],
    this: tx.Any,
    value: tx.Any,
) -> "MagicError":
    """Reconstruct a [`MagicError`][] from its undecorated parts."""
    # Module-level (rather than a lambda or a method) so that it can be
    # pickled by reference.
    return cls(message, *args, this=this, value=value)


class MagicError(Exception):
    """An exception raised by magic objects (factories, converters)."""

    def __new__(cls, *args, **kwargs) -> tx.Self:
        # Avoids errors in python 3.8, where Exception implements its own
        # __new__ but without keyword arguments, making subclasses fail
        # when they are initialized with keyword arguments.
        return super().__new__(cls, *args)

    def __init__(self, *args, **kwargs) -> None:
        """
        Other Parameters
        ----------------
        this : MagicHint
            The MagicHint instance that raised the error.
        value : Any
            The value that caused the error.
        """
        this = kwargs.pop("this", None)
        value = kwargs.pop("value", UNSET)
        self.this = this
        self.value = value
        if args:
            msg, *args = args
        else:
            msg = ""
        self.message = msg
        msg = self._make_message(msg, this=True, value=True, causes=False)
        super().__init__(msg, *args)

    @property
    def nice_message(self) -> str:
        return getattr(self, "args", ("",))[0]

    def __reduce__(self) -> tx.Tuple[tx.Any, ...]:
        # The default `BaseException.__reduce__` returns `(cls, self.args)`,
        # whose first element is the *decorated* message - so a round-trip
        # would decorate it a second time - and it drops `this`/`value`,
        # which live outside `args`. Rebuild from the undecorated parts.
        rest = tuple(self.args[1:])
        state = (type(self), self.message, rest, self.this, self.value)
        return (_rebuild_magic_error, state)

    @property
    def causes(self) -> tx.Tuple[Exception, ...]:
        if hasattr(self, "__all_causes__"):
            return self.__all_causes__
        if self.__cause__ is not None:
            # A `MultipleCauses` wrapper is transparent: expose the causes
            # it carries, not the wrapper itself.
            cause = self.__cause__
            return getattr(cause, "__all_causes__", (cause,))
        return ()

    @property
    def depth(self) -> int:
        return 1 + max(
            (getattr(p, "depth", 0) for p in self.causes), default=0
        )

    @property
    def best_cause(self) -> tx.Optional[tx.Self]:
        return max(
            self.causes,
            key=lambda p: getattr(p, "depth", 0),
            default=None
        )

    def _make_message(
        self,
        message: tx.Optional[str] = None,
        this: bool = True,
        value: bool = True,
        causes: bool = True
    ) -> str:
        if message is None:
            # `self.message` is the undecorated text. Using `nice_message`
            # here (which is `args[0]`, already decorated by `__init__`)
            # would prefix and append a second time at every level.
            message = self.message or ""

        # Only decorate with what was actually supplied. A `MagicError`
        # raised without a `this`/`value` used to render them anyway, as
        # a literal "None: " prefix and a "|> value = <UNSET>" line.
        if this and self.this is not None:
            if message:
                message = f"{self.this!r}: {message}"
            else:
                message = f"{self.this!r}"

        if value and self.value is not UNSET:
            message = f"{message}\n|> value = {self.value!r}"

        if causes and self.causes:
            arrow = "?>" if len(self.causes) > 1 else "->"
            cause_value = len(self.causes) == 1
            for cause in self.causes:
                if hasattr(cause, "_make_message"):
                    cause_message = cause._make_message(
                        this=this, value=cause_value
                    )
                else:
                    cause_message = f"{type(cause).__name__}: {cause}"
                message = f"{message}\n{arrow} {cause_message}"
        return message
