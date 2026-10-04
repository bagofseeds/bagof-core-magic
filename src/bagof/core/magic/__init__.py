
__all__ = [
    "MagicError",
    "MagicHint",
    "MultipleCauses",
    "eq_safenan",
    "get_concrete_type",
    "get_default",
    "get_from_registry",
    "get_origin_uw",
    "get_args_uw",
    "safe_get_origin",
    "safe_get_args",
    "safe_isinstance",
    "safe_issubclass",
    "ishintstance",
    "issubhint",
    "issubclassable",
    "issubscriptable",
    "is_typeddict",
    "typeddict_required_keys",
    "type2hint",
    "unwrap",
    "Unset",
    "UNSET",
    "NoneType",
    "REAL_TYPES",
    "UNION_TYPES",
    "UnionType",
    "clear_pending",
    "defer",
    "find_name",
    "has_module",
    "is_forward_ref",
    "lazy_import",
    "pending",
    "resolve_pending",
]

# The definitions live in private submodules; this module only re-exports
# them. Submodules and what each imports from its siblings:
#   _compat      constants, `UNSET`                 -
#   _introspect  safe introspection helpers         _compat
#   _relation    `issubhint`, `ishintstance`        _compat, _introspect
#   _lazy        lazy registry keys                 -
#   _registry    `get_from_registry`, ...           _compat, _introspect, _lazy
#   _errors      `MagicError`, `MultipleCauses`     _compat
#   _hint        `MagicHint`                        all but _registry, _lazy
# The `_x as _x` imports keep private names that tests and sibling packages
# import from here (and old `MagicError` pickles) resolving.

# locals
from ._compat import (
    REAL_TYPES,
    UNION_TYPES,
    UNSET,
    NoneType,
    UnionType,
    Unset,
)
from ._compat import _is_special_form as _is_special_form
from ._errors import MagicError, MultipleCauses
from ._errors import _rebuild_magic_error as _rebuild_magic_error
from ._hint import MagicHint
from ._introspect import _is_concrete_type as _is_concrete_type
from ._introspect import _unwrap_typevar as _unwrap_typevar
from ._introspect import (
    eq_safenan,
    get_args_uw,
    get_concrete_type,
    get_origin_uw,
    is_typeddict,
    issubclassable,
    issubscriptable,
    safe_get_args,
    safe_get_origin,
    safe_isinstance,
    safe_issubclass,
    type2hint,
    typeddict_required_keys,
    unwrap,
)
from ._introspect import normalise_hint as normalise_hint
from ._lazy import (
    clear_pending,
    defer,
    find_name,
    has_module,
    is_forward_ref,
    lazy_import,
    pending,
    resolve_pending,
)
from ._registry import _type_dist as _type_dist
from ._registry import get_default, get_from_registry
from ._relation import _ishintstance_type as _ishintstance_type
from ._relation import _issubliteral as _issubliteral
from ._relation import _issubnone as _issubnone
from ._relation import _issubtype as _issubtype
from ._relation import _issubtypevar as _issubtypevar
from ._relation import _issubunion as _issubunion
from ._relation import ishintstance, issubhint
