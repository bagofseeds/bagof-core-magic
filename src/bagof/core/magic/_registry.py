"""
Lookups keyed by type hints:
[`get_from_registry`][bagof.core.magic.get_from_registry] and
[`get_default`][bagof.core.magic.get_default].

The public names are re-exported by [`bagof.core.magic`][]; import them
from there.
"""

__all__ = [
    "get_default",
    "get_from_registry",
]

# stdlib
from collections import abc

# dependencies
import typing_extensions as tx

# locals
from ._compat import UNION_TYPES, UNSET, NoneType
from ._introspect import (
    _TYPE2HINT,
    _all_orig_bases,
    _canonical_typeddict,
    is_typeddict,
    issubclassable,
    normalise_hint,
    safe_get_args,
    safe_get_origin,
    safe_isinstance,
    safe_issubclass,
    unwrap,
)
from ._lazy import resolve_pending


def get_default(hint: tx.Any) -> tx.Any:
    """
    Get a default value from a type hint.

    * If the hint is a [`Literal`][tx.Literal], the first value in the
      literal is returned ([`None`][], if [`None`][] is one of the
      literal's values).
    * If the hint is a [`Union`][tx.Union] that contains
      [`NoneType`][types.NoneType], [`None`][] is returned.
    * Otherwise, if the hint is a [`Union`][tx.Union], we recurse through
      its sub-hints and return the first default found.
    * If no default value can be found, a [`TypeError`][] is raised.
      A factory should then be used.
    """
    origin = safe_get_origin(hint, unwrap=tx.Annotated)
    args = safe_get_args(hint, unwrap=tx.Annotated)
    if origin is tx.Literal and args:
        if None in args:
            return None
        return args[0]
    if origin in UNION_TYPES:
        if NoneType in args:
            return None
        for arg in args:
            try:
                return get_default(arg)
            except TypeError:
                continue
    raise TypeError(f"Cannot get default for hint {hint}")


def get_from_registry(hint: tx.Any, registry: dict) -> tx.Any:
    """
    Get the best matching value from a registry whose keys are types or
    type hints.

    The best match is the registry key that is the narrowest superclass
    (or superhint) of `hint`, following its MRO; exact matches are always
    preferred. If `hint` is [`Annotated`][typing.Annotated] and no match is
    found for it directly, the search is retried against its unwrapped
    hint.

    Exact matches are by hint *equality*, so `List[int]` and `list[int]`
    are different keys, but `Union[int, str]` and `Union[str, int]` are the
    same one.

    Pending lazy keys (see [`defer`][]) whose target has been imported
    since are moved into their registry first, so every caller sees them.

    Two keys at the same distance from `hint` are told apart by
    specificity (a subclass key beats its superclass), and then by
    registry order. Distances along an MRO are all different, so a true
    tie only happens between keys `hint` is a *virtual* subclass of (such
    as two unrelated ABCs it is registered with). A lazy key enters the
    registry when it resolves, so that is the position it ties at.

    !!! example
        ```pycon
        >>> registry = {int: "number", object: "any"}
        >>> get_from_registry(bool, registry)
        'number'
        >>> get_from_registry(str, registry)
        'any'
        ```
    """
    resolve_pending()

    # A bare `None` means `NoneType` as a hint, so it is matched as one --
    # the same normalisation a `MagicHint` built from it would apply.
    hint = normalise_hint(hint)

    # Exact-identity pass, before any origin is taken. `_get_best_match`
    # compares origins, which erases what a Union/Literal/TypeVar or a
    # parameterised generic actually is -- so a registry key that is one of
    # those is only reachable here, though the "exact matches preferred"
    # promise above is meant to hold for every hint.
    match = _exact_from_registry(hint, registry)
    if match is not UNSET:
        return match

    # First naive pass
    best_match, best_dist = _get_best_match(hint, registry)

    # Second pass, where Annotated hints are unwrapped. First for an exact
    # key the inner hint is (a specific Union/... carried under metadata),
    # then for a better origin match. Only used if it beats the first pass.
    if best_dist != 0 and safe_get_origin(hint) is tx.Annotated:
        inner = unwrap(hint, tx.Annotated)
        match = _exact_from_registry(inner, registry)
        if match is not UNSET:
            return match
        better_match, better_dist = _get_best_match(inner, registry)
        if better_dist < best_dist:
            best_match, best_dist = better_match, better_dist

    if best_match is not None:
        return registry[best_match]

    return None


def _exact_from_registry(hint: tx.Any, registry: dict) -> tx.Any:
    """The value `hint` is registered under by equality, or `UNSET`.

    A `Union`/`Literal` compares order-insensitively and a `TypeVar` by
    identity, which is exactly the "same hint" test wanted. A new-style
    generic (`list[int]`) is also tried in its `typing` spelling
    (`List[int]`), so a registry keyed one way is reached by a query
    written the other. A hint that cannot be hashed -- `Annotated` with
    mutable metadata, `Literal[[...]]`, or a bare metadata object the bags
    pass straight in -- is simply not an exact key, so the lookup falls
    through rather than raising.
    """
    for candidate in (hint, _typing_spelling(hint)):
        try:
            if candidate in registry:
                return registry[candidate]
        except TypeError:
            pass
    return UNSET


def _get_best_match(hint: tx.Any, registry: dict) -> tx.Tuple[tx.Any, float]:
    """
    Get the best matching value from a registry whose keys are types or
    type hints, and return the key and value as a tuple.
    """
    hint = safe_get_origin(hint)

    best_match, best_dist = None, float("inf")
    for key in registry:

        dist = _type_dist(hint, key)

        if dist == float("inf"):
            # Not a match at all.
            continue

        if dist == 0:
            # Perfect match -> stop here
            best_match, best_dist = key, dist
            break

        if best_match is None:
            best_match, best_dist = key, dist
            continue

        key_is_td, best_is_td = is_typeddict(key), is_typeddict(best_match)
        if key_is_td != best_is_td:
            # Prefer a typeddict key over a plain one. Their distances are
            # measured along *different* hierarchies -- `__orig_bases__`
            # for a typeddict, `__mro__` for a class -- so the two numbers
            # are not comparable, and the nearer one is not the better
            # one. A typeddict inheriting from another typeddict sits at
            # distance 2 from `TypedDict` but only 1 from `dict`, and used
            # to be handed to the `dict` entry.
            if key_is_td:
                best_match, best_dist = key, dist
            continue

        if dist < best_dist:
            best_match, best_dist = key, dist

        elif dist == best_dist and safe_issubclass(key, best_match):
            # Prefer more specific subclass
            best_match = key

    return best_match, best_dist


def _type_dist(subcls: type, cls: type) -> float:
    """Distance between two types, based on their inheritance hierarchy."""
    if safe_isinstance(subcls, tx.TypeVar):
        subcls = tx.TypeVar
    if subcls is cls:
        return 0
    if not issubclassable(subcls) or not issubclassable(cls):
        return float("inf")
    if not safe_issubclass(subcls, cls):
        return float("inf")
    # Our `is_typeddict`, not `tx.is_typeddict`: the latter is False for
    # `TypedDict` itself, which would send a typeddict subclass down the
    # `__mro__` branch, where `TypedDict` never appears - so the loop below
    # would fall through and report the "not found" distance instead of 1.
    if is_typeddict(cls):
        cls = _canonical_typeddict(cls)
        bases = _all_orig_bases(subcls)
    else:
        bases = subcls.__mro__
    distance = 0
    for base in bases:
        if base is cls:
            return distance
        distance += 1
    return 1000


def _typing_spelling(hint: tx.Any) -> tx.Any:
    """`list[int]` rewritten as `List[int]`, recursively; else unchanged.

    A parameterised builtin or abc generic (`list[int]`, `dict[str, int]`)
    is a different object from its `typing` twin (`List[int]`,
    `Dict[str, int]`) and does not compare equal to it, so a registry keyed
    one way misses a query written the other. Rewriting the new-style form
    into the `typing` spelling lets the two meet.

    The rewrite reaches all the way down, so a new-style generic nested
    inside a `Union`, `Optional`, `Annotated`, `Callable`, or another
    generic (`Optional[list[int]]`, `Callable[[list[int]], str]`) is
    rewritten too. `Literal` is left alone: its arguments are values, not
    types. It rewrites the query only, so it reaches a registry keyed in
    the `typing` spelling from a new-style query, not the other way round.
    `list[int]` does not exist before Python 3.9, so there is nothing to
    rewrite there.
    """
    origin = tx.get_origin(hint)
    if origin is None or origin is tx.Literal:
        return hint
    args = tx.get_args(hint)
    try:
        if not args:
            # `tuple[()]` reports no arguments from Python 3.11 on, but it
            # is the empty-tuple type and still differs from `Tuple[()]`.
            return tx.Tuple[()] if origin is tuple else hint
        if origin is tx.Annotated:
            # `(type, *metadata)`: rewrite the type, keep the metadata.
            inner = _typing_spelling(args[0])
            if inner == args[0]:
                return hint
            return tx.Annotated[(inner, *args[1:])]
        if origin is abc.Callable and len(args) == 2:
            # `(parameters, return)`: the parameters are a list of types,
            # or `...` / a `ParamSpec`, which are left whole.
            params, ret = args
            if isinstance(params, list):
                params = [_typing_spelling(each) for each in params]
            return tx.Callable[(params, _typing_spelling(ret))]
        spelled = tuple(_typing_spelling(arg) for arg in args)
        if origin in UNION_TYPES:
            return tx.Union[spelled]
        # A builtin/abc container gets its `typing` spelling; any other
        # generic (a user `Generic`) is rebuilt on its own origin.
        typing_origin = _TYPE2HINT.get(origin, origin)
        return typing_origin[spelled if len(spelled) > 1 else spelled[0]]
    except Exception:
        # A rebuild that fails -- a user origin that refuses these
        # arguments, an exotic `Callable` form -- leaves the hint as it
        # was, to be matched by its origin instead.
        return hint
