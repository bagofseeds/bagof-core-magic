# bagof-core-magic

Core tools used by magic converters, validators, factories, etc.

It provides the type-hint introspection machinery shared by those
packages - functions such as `issubhint`, `ishintstance`, `unwrap`, and
`get_default` for inspecting and comparing `typing` hints at runtime -
along with the `MagicHint` base class and `MagicError` exception that
magic objects are built on.

It also lets registries (the dicts `get_from_registry` searches) be keyed
lazily: `defer(registry, ForwardRef("dask.array.Array"), value, context)`
registers `value` once someone else has imported `dask.array`, without
ever importing it - so a package can register values for optional
libraries without paying their import time. `lazy_import` and
`has_module` help define such classes without importing the library.
