"""Import smoke tests, and lazy registry keys in a clean process (#75)."""

# stdlib
import importlib
import pathlib
import subprocess
import sys
import textwrap

# dependencies
import pytest

BACKENDS = ("dask", "dask.array", "cupy", "pandas")


def _run(code: str, *path: str) -> str:
    """Run `code` in a fresh interpreter, with `path` prepended."""
    prelude = f"import sys; sys.path[:0] = {list(map(str, path))!r}\n"
    result = subprocess.run(
        [sys.executable, "-c", prelude + textwrap.dedent(code)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout


@pytest.fixture
def lazypkg(tmp_path: pathlib.Path) -> pathlib.Path:
    """A package whose import is recorded, with a lazy-loading submodule."""
    pkg = tmp_path / "_bagof_lazypkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text(textwrap.dedent("""
        import importlib
        import sys
        sys.modules["_bagof_lazy_log"] = log = []
        log.append(__name__)

        def __getattr__(name):
            # PEP 562: imports the submodule on attribute access
            return importlib.import_module(f"{__name__}.{name}")
    """))
    (pkg / "sub.py").write_text(textwrap.dedent("""
        import sys
        sys.modules["_bagof_lazy_log"].append(__name__)

        class Thing:
            pass
    """))
    return tmp_path


def test_template_submodule_is_importable() -> None:
    """The template package should be importable after installation."""
    module = importlib.import_module("bagof.core.magic")
    assert module is not None


def test_import_does_not_import_backends() -> None:
    out = _run(f"""
        import bagof.core.magic
        print(sorted(m for m in {BACKENDS!r} if m in sys.modules))
    """)
    assert out.strip() == "[]"


def test_lazy_key_does_not_import_its_target(lazypkg: pathlib.Path) -> None:
    out = _run("""
        import typing
        from bagof.core.magic import (
            defer, get_from_registry, has_module, lazy_import, pending,
        )

        registry = {object: "any"}
        defer(registry, typing.ForwardRef("_bagof_lazypkg.sub.Thing"), "lazy")

        class Holder:
            CLS = lazy_import("_bagof_lazypkg.sub.Thing")

        assert has_module("_bagof_lazypkg.sub")
        assert get_from_registry(int, registry) == "any"
        assert pending(registry), pending(registry)
        assert "_bagof_lazypkg" not in sys.modules

        # the parent alone does not resolve the key, nor import the
        # submodule through its PEP 562 `__getattr__`
        import _bagof_lazypkg
        assert get_from_registry(int, registry) == "any"
        assert "_bagof_lazypkg.sub" not in sys.modules

        # someone else imports the target: the key resolves
        from _bagof_lazypkg.sub import Thing
        assert get_from_registry(Thing, registry) == "lazy"
        assert registry[Thing] == "lazy"
        assert Holder.CLS is Thing
        print(sys.modules["_bagof_lazy_log"])
    """, lazypkg)
    assert out.strip() == "['_bagof_lazypkg', '_bagof_lazypkg.sub']"


def test_lazy_import_imports_on_access(lazypkg: pathlib.Path) -> None:
    out = _run("""
        from bagof.core.magic import lazy_import

        class Holder:
            CLS = lazy_import("_bagof_lazypkg.sub.Thing")

        assert "_bagof_lazypkg" not in sys.modules
        print(Holder.CLS.__module__)
    """, lazypkg)
    assert out.strip() == "_bagof_lazypkg.sub"


def test_dask_key_resolves_after_late_import() -> None:
    pytest.importorskip("dask.array")
    out = _run("""
        import typing
        from bagof.core.magic import defer, get_from_registry

        registry = {}
        defer(registry, typing.ForwardRef("dask.array.Array"), "dask")
        assert "dask" not in sys.modules
        import dask.array as da
        print(get_from_registry(da.Array, registry))
    """)
    assert out.strip() == "dask"
