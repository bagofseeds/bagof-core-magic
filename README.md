# bagof-core-magic

Core tools used by magic converters, validators, factories, etc.

It provides the type-hint introspection machinery shared by those
packages - functions such as `issubhint`, `ishintstance`, `unwrap`, and
`get_default` for inspecting and comparing `typing` hints at runtime -
along with the `MagicHint` base class and `MagicError` exception that
magic objects are built on.
