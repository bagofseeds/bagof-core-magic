"""Tests for lazy (forward-reference) registry keys (#75)."""

# stdlib
import gc
import sys
import threading
import types

# dependencies
import pytest
import typing_extensions as tx

# locals
import bagof.core.magic as magic
from bagof.core.magic import (
    clear_pending,
    defer,
    find_name,
    get_from_registry,
    has_module,
    is_forward_ref,
    lazy_import,
    pending,
    resolve_pending,
)


@pytest.fixture(autouse=True)
def _restore_pending() -> tx.Iterator[None]:
    """Drop the pending entries a test leaves behind."""
    saved = list(magic._PENDING)
    yield
    magic._PENDING[:] = saved


def _fake_module(name: str, monkeypatch: tx.Any) -> tx.Any:
    module = types.ModuleType(name)
    monkeypatch.setitem(sys.modules, name, module)
    return module


class _Thing:
    pass


class _Other:
    pass


# --- the public surface ------------------------------------------------


@pytest.mark.parametrize(
    "name",
    ["clear_pending", "defer", "find_name", "has_module", "is_forward_ref",
     "lazy_import", "pending", "resolve_pending"],
)
def test_lazy_names_are_exported(name: str) -> None:
    assert name in magic.__all__


def test_is_forward_ref() -> None:
    assert is_forward_ref(tx.ForwardRef("x.Y"))
    assert not is_forward_ref("x.Y")
    assert not is_forward_ref(int)


def test_defer_rejects_a_plain_string() -> None:
    # Only `ForwardRef` keys are lazy; a string is an ordinary key.
    with pytest.raises(TypeError):
        defer({}, "a.b", 1, None)


# --- resolution ---------------------------------------------------------


def test_forward_ref_key_waits_for_its_module(monkeypatch: tx.Any) -> None:
    registry: tx.Dict[tx.Any, tx.Any] = {}
    defer(registry, tx.ForwardRef("_bagof_lazy_mod.Thing"), "lazy")
    assert "_bagof_lazy_mod" not in sys.modules
    assert registry == {}
    assert pending(registry) == [(None, "_bagof_lazy_mod.Thing", "lazy")]

    _fake_module("_bagof_lazy_mod", monkeypatch).Thing = _Thing
    assert get_from_registry(_Thing, registry) == "lazy"
    assert registry == {_Thing: "lazy"}
    assert pending(registry) == []


def test_forward_ref_key_resolvable_now_is_registered_at_once(
    monkeypatch: tx.Any,
) -> None:
    _fake_module("_bagof_lazy_now", monkeypatch).Thing = _Thing
    registry: tx.Dict[tx.Any, tx.Any] = {}
    defer(registry, tx.ForwardRef("_bagof_lazy_now.Thing"), "lazy")
    assert registry == {_Thing: "lazy"}
    assert pending(registry) == []


def test_forward_ref_key_of_a_partial_module_stays_pending(
    monkeypatch: tx.Any,
) -> None:
    """A module still initialising (name not defined yet) is retried."""
    registry: tx.Dict[tx.Any, tx.Any] = {}
    module = _fake_module("_bagof_lazy_partial", monkeypatch)
    defer(registry, tx.ForwardRef("_bagof_lazy_partial.Thing"), "lazy")
    assert registry == {}

    module.Thing = _Thing
    assert get_from_registry(_Thing, registry) == "lazy"


def test_forward_ref_key_to_a_nested_class(monkeypatch: tx.Any) -> None:
    """The part after the module is walked as attributes."""

    class Outer:
        class Inner:
            pass

    registry: tx.Dict[tx.Any, tx.Any] = {}
    _fake_module("_bagof_lazy_nested", monkeypatch).Outer = Outer
    defer(registry, tx.ForwardRef("_bagof_lazy_nested.Outer.Inner"), "lazy")
    assert registry == {Outer.Inner: "lazy"}


def test_forward_ref_key_of_an_unloaded_submodule_stays_pending(
    monkeypatch: tx.Any,
) -> None:
    """A loaded parent package does not resolve an unloaded submodule."""
    registry: tx.Dict[tx.Any, tx.Any] = {}
    _fake_module("_bagof_lazy_pkg", monkeypatch)
    defer(registry, tx.ForwardRef("_bagof_lazy_pkg.sub.Thing"), "lazy")
    assert get_from_registry(_Thing, registry) is None
    assert registry == {}

    # the longest loaded prefix is used once the submodule is imported
    _fake_module("_bagof_lazy_pkg.sub", monkeypatch).Thing = _Thing
    assert get_from_registry(_Thing, registry) == "lazy"


def test_module_getattr_is_never_called(monkeypatch: tx.Any) -> None:
    """A PEP 562 `__getattr__` (which may import) is bypassed."""
    calls = []

    def __getattr__(name: str) -> tx.Any:
        calls.append(name)
        return _Thing

    _fake_module("_bagof_lazy_pep562", monkeypatch).__getattr__ = __getattr__
    registry: tx.Dict[tx.Any, tx.Any] = {}
    defer(registry, tx.ForwardRef("_bagof_lazy_pep562.Thing"), "lazy")
    assert get_from_registry(_Thing, registry) is None
    with pytest.raises(NameError):
        find_name("_bagof_lazy_pep562.Thing")
    assert calls == []


def test_forward_ref_key_relative_to_the_context_module(
    monkeypatch: tx.Any,
) -> None:
    """A bare name is looked up in the context module."""
    registry: tx.Dict[tx.Any, tx.Any] = {}
    module = _fake_module("_bagof_lazy_rel", monkeypatch)
    defer(registry, tx.ForwardRef("MyThing"), "lazy", "_bagof_lazy_rel")
    assert registry == {}
    # defined later in the module (e.g. after the registering class)
    module.MyThing = _Thing
    assert get_from_registry(_Thing, registry) == "lazy"


def test_forward_ref_key_relative_alias_of_a_runtime_import(
    monkeypatch: tx.Any,
) -> None:
    """`"np.Thing"` follows the module's own runtime `import ... as np`."""
    registry: tx.Dict[tx.Any, tx.Any] = {}
    target = _fake_module("_bagof_lazy_target", monkeypatch)
    target.Thing = _Thing
    _fake_module("_bagof_lazy_alias", monkeypatch).np = target
    defer(registry, tx.ForwardRef("np.Thing"), "lazy", "_bagof_lazy_alias")
    assert registry == {_Thing: "lazy"}


@pytest.mark.skipif(
    sys.version_info < (3, 9, 7), reason="ForwardRef(module=) is 3.9.7+"
)
def test_forward_ref_key_honours_module(monkeypatch: tx.Any) -> None:
    """`ForwardRef(name, module=...)` takes precedence over the context."""
    registry: tx.Dict[tx.Any, tx.Any] = {}
    _fake_module("_bagof_lazy_elsewhere", monkeypatch).Thing = _Other
    ref = tx.ForwardRef("Thing", module="_bagof_lazy_modarg")
    defer(registry, ref, "lazy", "_bagof_lazy_elsewhere")
    assert registry == {}
    _fake_module("_bagof_lazy_modarg", monkeypatch).Thing = _Thing
    assert get_from_registry(_Thing, registry) == "lazy"
    assert registry == {_Thing: "lazy"}


def test_forward_ref_key_type_checking_alias_stays_pending(
    monkeypatch: tx.Any,
) -> None:
    """An alias that only exists for type checkers cannot be resolved."""
    registry: tx.Dict[tx.Any, tx.Any] = {}
    _fake_module("_bagof_lazy_tc", monkeypatch)  # no runtime `xx` alias
    _fake_module("_bagof_lazy_real", monkeypatch).Thing = _Thing

    defer(registry, tx.ForwardRef("xx.Thing"), "lazy", "_bagof_lazy_tc")
    assert registry == {}
    assert pending(registry) == [("_bagof_lazy_tc", "xx.Thing", "lazy")]

    # the absolute dotted path resolves
    defer(
        registry, tx.ForwardRef("_bagof_lazy_real.Thing"), "lazy",
        "_bagof_lazy_tc",
    )
    assert registry == {_Thing: "lazy"}


def test_forward_ref_key_relative_wins_over_absolute(
    monkeypatch: tx.Any,
) -> None:
    """A module global shadows a top-level module of the same name."""
    registry: tx.Dict[tx.Any, tx.Any] = {}
    _fake_module("_bagof_lazy_clash", monkeypatch).Thing = _Other
    local = _fake_module("_bagof_lazy_clash_ctx", monkeypatch)
    relative = _fake_module("_bagof_lazy_clash_rel", monkeypatch)
    relative.Thing = _Thing
    local._bagof_lazy_clash = relative

    defer(
        registry, tx.ForwardRef("_bagof_lazy_clash.Thing"), "lazy",
        "_bagof_lazy_clash_ctx",
    )
    assert registry == {_Thing: "lazy"}


def test_forward_ref_key_relative_miss_falls_back_to_absolute(
    monkeypatch: tx.Any,
) -> None:
    """A relative first part whose walk fails still tries the absolute."""
    registry: tx.Dict[tx.Any, tx.Any] = {}
    _fake_module("_bagof_lazy_fb", monkeypatch).Thing = _Thing
    # the context has a `_bagof_lazy_fb` global without `Thing`
    _fake_module("_bagof_lazy_fb_ctx", monkeypatch)._bagof_lazy_fb = object()
    defer(
        registry, tx.ForwardRef("_bagof_lazy_fb.Thing"), "lazy",
        "_bagof_lazy_fb_ctx",
    )
    assert registry == {_Thing: "lazy"}


def test_find_name_of_an_unknown_name_raises() -> None:
    with pytest.raises(NameError):
        find_name("_bagof_lazy_nowhere.Thing")
    with pytest.raises(NameError, match="_bagof_lazy_ctx"):
        find_name("Thing", "_bagof_lazy_ctx")


def test_find_name_of_a_known_name(monkeypatch: tx.Any) -> None:
    _fake_module("_bagof_lazy_known", monkeypatch).Thing = _Thing
    assert find_name("_bagof_lazy_known.Thing") is _Thing
    assert find_name("Thing", "_bagof_lazy_known") is _Thing


def test_builtins_are_not_looked_up(monkeypatch: tx.Any) -> None:
    _fake_module("_bagof_lazy_builtin", monkeypatch)
    registry: tx.Dict[tx.Any, tx.Any] = {}
    defer(registry, tx.ForwardRef("int"), "lazy", "_bagof_lazy_builtin")
    assert registry == {}
    assert len(pending(registry)) == 1


@pytest.mark.parametrize(
    "name", ["Optional[Foo]", "a | None", "a..b", "1a.B", "", "a.b()"]
)
def test_defer_rejects_a_non_dotted_name(name: str) -> None:
    # Some of these are refused by `ForwardRef` itself (it compiles its
    # argument) on some Pythons; build the ref regardless.
    ref = tx.ForwardRef("x")
    ref.__forward_arg__ = name
    with pytest.raises(ValueError, match="dotted name"):
        defer({}, ref, 1)


# --- ordering -----------------------------------------------------------


def test_forward_ref_key_and_real_key_last_registration_wins(
    monkeypatch: tx.Any,
) -> None:
    """Forward-ref and real keys for one object keep "last one wins"."""
    registry: tx.Dict[tx.Any, tx.Any] = {}

    # ref, then ref: the second replaces the first while pending (the
    # refs compare by name, whatever else `ForwardRef` equality involves)
    ref = "_bagof_lazy_order.Thing"
    defer(registry, tx.ForwardRef(ref), "first")
    defer(registry, tx.ForwardRef(ref), "second")
    assert pending(registry) == [(None, ref, "second")]
    _fake_module("_bagof_lazy_order", monkeypatch).Thing = _Thing

    # real key written after: settling the pending ones first lets it win
    resolve_pending()
    registry[_Thing] = "third"
    assert get_from_registry(_Thing, registry) == "third"

    # real, then ref (module loaded): the ref key wins
    defer(registry, tx.ForwardRef(ref), "fourth")
    assert get_from_registry(_Thing, registry) == "fourth"


def test_older_lazy_key_does_not_override_a_newer_plain_key(
    monkeypatch: tx.Any,
) -> None:
    """A plain key written once the target exists beats a pending one."""
    registry: tx.Dict[tx.Any, tx.Any] = {}
    defer(registry, tx.ForwardRef("_bagof_lazy_user.Thing"), "package")
    # the user imports the library and registers their own value, without
    # any lookup in between
    _fake_module("_bagof_lazy_user", monkeypatch).Thing = _Thing
    registry[_Thing] = "user"
    assert get_from_registry(_Thing, registry) == "user"
    assert pending(registry) == []


def test_two_spellings_resolved_in_one_pass_keep_registration_order(
    monkeypatch: tx.Any,
) -> None:
    registry: tx.Dict[tx.Any, tx.Any] = {}
    defer(registry, tx.ForwardRef("_bagof_lazy_pass.Thing"), "older")
    defer(registry, tx.ForwardRef("_bagof_lazy_pass.Alias"), "newer")
    module = _fake_module("_bagof_lazy_pass", monkeypatch)
    module.Thing = module.Alias = _Thing
    assert get_from_registry(_Thing, registry) == "newer"


def test_two_spellings_resolved_in_two_passes_first_wins(
    monkeypatch: tx.Any,
) -> None:
    """Documented: across passes, the first spelling to resolve wins."""
    registry: tx.Dict[tx.Any, tx.Any] = {}
    defer(registry, tx.ForwardRef("_bagof_lazy_late.Thing"), "older")
    defer(registry, tx.ForwardRef("_bagof_lazy_early.Thing"), "newer")
    _fake_module("_bagof_lazy_early", monkeypatch).Thing = _Thing
    assert get_from_registry(_Thing, registry) == "newer"
    _fake_module("_bagof_lazy_late", monkeypatch).Thing = _Thing
    assert get_from_registry(_Thing, registry) == "newer"
    assert pending(registry) == []


def test_clear_pending(monkeypatch: tx.Any) -> None:
    one: tx.Dict[tx.Any, tx.Any] = {}
    two: tx.Dict[tx.Any, tx.Any] = {}
    defer(one, tx.ForwardRef("_bagof_lazy_clear.Thing"), 1)
    defer(two, tx.ForwardRef("_bagof_lazy_clear.Thing"), 2)
    clear_pending(one)
    assert pending(one) == []
    assert pending(two) == [(None, "_bagof_lazy_clear.Thing", 2)]
    clear_pending()
    assert pending() == []
    # nothing is applied
    _fake_module("_bagof_lazy_clear", monkeypatch).Thing = _Thing
    resolve_pending()
    assert one == two == {}


def test_defer_settles_older_entries_first(monkeypatch: tx.Any) -> None:
    """An older spelling resolvable now does not override a newer one."""
    registry: tx.Dict[tx.Any, tx.Any] = {}
    defer(registry, tx.ForwardRef("_bagof_lazy_seq.Thing"), "older")
    module = _fake_module("_bagof_lazy_seq", monkeypatch)
    module.Thing = _Thing
    module.Alias = _Thing
    defer(registry, tx.ForwardRef("_bagof_lazy_seq.Alias"), "newer")
    assert get_from_registry(_Thing, registry) == "newer"


def test_pending_entries_are_per_registry(monkeypatch: tx.Any) -> None:
    one: tx.Dict[tx.Any, tx.Any] = {}
    two: tx.Dict[tx.Any, tx.Any] = {}
    ref = "_bagof_lazy_two.Thing"
    defer(one, tx.ForwardRef(ref), 1)
    defer(two, tx.ForwardRef(ref), 2)
    assert pending(one) == [(None, ref, 1)]
    assert pending(two) == [(None, ref, 2)]
    assert (None, ref, 1) in pending() and (None, ref, 2) in pending()

    _fake_module("_bagof_lazy_two", monkeypatch).Thing = _Thing
    # one lookup resolves every registry
    assert get_from_registry(_Thing, one) == 1
    assert two == {_Thing: 2}


def test_resolve_pending_returns_at_once_when_nothing_is_pending(
    monkeypatch: tx.Any,
) -> None:
    magic._PENDING[:] = []

    def boom(*args: tx.Any) -> tx.NoReturn:
        raise AssertionError("looked up a name with nothing pending")

    monkeypatch.setattr(magic, "_find_name", boom)
    resolve_pending()
    assert get_from_registry(int, {int: 1}) == 1


def test_lazy_key_takes_part_in_subclass_matching(
    monkeypatch: tx.Any,
) -> None:
    class Sub(_Thing):
        pass

    registry: tx.Dict[tx.Any, tx.Any] = {object: "any"}
    defer(registry, tx.ForwardRef("_bagof_lazy_mro.Thing"), "thing")
    assert get_from_registry(Sub, registry) == "any"
    _fake_module("_bagof_lazy_mro", monkeypatch).Thing = _Thing
    assert get_from_registry(Sub, registry) == "thing"


# --- lifetime and failure modes ----------------------------------------


def test_pending_entry_of_a_collected_registry_is_dropped() -> None:
    class Registry(dict):
        pass

    registry = Registry()
    defer(registry, tx.ForwardRef("_bagof_lazy_gc.Thing"), "lazy")
    assert len(pending(registry)) == 1
    count = len(magic._PENDING)
    del registry
    gc.collect()
    assert all(
        name != "_bagof_lazy_gc.Thing" for _, name, _ in pending()
    )
    resolve_pending()
    assert len(magic._PENDING) == count - 1


def test_plain_dict_registry_is_held_until_resolved(
    monkeypatch: tx.Any,
) -> None:
    registry: tx.Dict[tx.Any, tx.Any] = {}
    defer(registry, tx.ForwardRef("_bagof_lazy_strong.Thing"), "lazy")
    entry = magic._PENDING[-1]
    assert entry.registry() is registry


def test_unhashable_target_is_dropped_with_a_warning(
    monkeypatch: tx.Any,
) -> None:
    registry: tx.Dict[tx.Any, tx.Any] = {}
    defer(registry, tx.ForwardRef("_bagof_lazy_unhash.Thing"), "lazy")
    _fake_module("_bagof_lazy_unhash", monkeypatch).Thing = []
    with pytest.warns(RuntimeWarning, match="unhashable"):
        assert get_from_registry(int, registry) is None
    assert pending(registry) == []
    assert registry == {}


def test_concurrent_resolution_applies_each_entry_once(
    monkeypatch: tx.Any,
) -> None:
    class Registry(dict):
        writes = 0

        def __setitem__(self, key: tx.Any, value: tx.Any) -> None:
            type(self).writes += 1
            super().__setitem__(key, value)

    registry = Registry()
    names = [f"Thing{i}" for i in range(50)]
    for name in names:
        defer(registry, tx.ForwardRef(f"_bagof_lazy_mt.{name}"), name)
    module = _fake_module("_bagof_lazy_mt", monkeypatch)
    for name in names:
        setattr(module, name, type(name, (), {}))

    barrier = threading.Barrier(8)

    def worker() -> None:
        barrier.wait()
        for _ in range(20):
            resolve_pending()

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert Registry.writes == len(names)
    assert sorted(registry.values()) == sorted(names)
    assert pending(registry) == []


# --- helpers ------------------------------------------------------------


def test_has_module() -> None:
    assert has_module("typing_extensions")
    assert not has_module("_bagof_lazy_not_installed")
    # only the top-level name is looked up
    assert has_module("typing_extensions._bagof_no_such_submodule")


def test_lazy_import_imports_on_first_access(monkeypatch: tx.Any) -> None:
    module = types.ModuleType("_bagof_lazy_imp")
    module.Thing = _Thing
    imported = []

    def fake_import(name: str) -> tx.Any:
        imported.append(name)
        return {"_bagof_lazy_imp": module}[name]

    monkeypatch.setattr(magic.importlib, "import_module", fake_import)

    class Holder:
        DEFAULT = lazy_import("_bagof_lazy_imp.Thing")

    assert imported == []
    assert Holder.DEFAULT is _Thing
    assert Holder().DEFAULT is _Thing
    assert imported == ["_bagof_lazy_imp"]
    assert "_bagof_lazy_imp.Thing" in repr(Holder.__dict__["DEFAULT"])


def test_lazy_import_of_a_submodule() -> None:
    class Holder:
        CLS = lazy_import("email.mime.text.MIMEText")

    from email.mime.text import MIMEText

    assert Holder.CLS is MIMEText


def test_lazy_import_of_a_missing_class_attribute_raises() -> None:
    class Holder:
        CLS = lazy_import("collections.OrderedDict._bagof_missing")

    with pytest.raises(AttributeError):
        Holder.CLS  # noqa: B018
