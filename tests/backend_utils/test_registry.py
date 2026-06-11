from __future__ import annotations

import pytest

from pyrpoc.backend_utils.registry import Registry


class Base:
    display_name = "Base"


class Alpha(Base):
    display_name = "Alpha Widget"
    parameter_groups = {"group": []}

    def __init__(self, value: int = 0):
        self.value = value

    @classmethod
    def get_contract(cls):
        return {"key": "alpha"}


class Beta(Base):
    pass


class Unrelated:
    pass


@pytest.fixture
def registry() -> Registry:
    reg = Registry("things", Base)
    reg.register("alpha")(Alpha)
    reg.register("beta")(Beta)
    return reg


def test_register_and_list_keys_sorted(registry):
    assert registry.list_keys() == ["alpha", "beta"]
    assert registry.get_registered() == ["alpha", "beta"]


def test_get_class(registry):
    assert registry.get_class("alpha") is Alpha


def test_get_class_unknown_raises(registry):
    with pytest.raises(KeyError):
        registry.get_class("missing")


def test_create_passes_kwargs(registry):
    instance = registry.create("alpha", value=7)
    assert isinstance(instance, Alpha)
    assert instance.value == 7


def test_register_rejects_non_subclass():
    reg = Registry("things", Base)
    with pytest.raises(TypeError):
        reg.register("bad")(Unrelated)


def test_register_rejects_duplicate_key():
    reg = Registry("things", Base)
    reg.register("alpha")(Alpha)
    with pytest.raises(KeyError):
        reg.register("alpha")(Beta)


def test_describe_uses_contract_attributes(registry):
    described = registry.describe("alpha")
    assert described["key"] == "alpha"
    assert described["class_name"] == "Alpha"
    assert described["display_name"] == "Alpha Widget"
    assert described["parameters"] == {"group": []}
    assert described["contract"] == {"key": "alpha"}


def test_describe_falls_back_to_class_name(registry):
    # Beta declares no display_name override beyond the inherited "Base" string,
    # and no parameter_groups, so defaults are returned.
    described = registry.describe("beta")
    assert described["parameters"] == {}
    assert described["config_parameters"] == {}
    assert described["actions"] == []
    assert described["contract"] == {}


def test_describe_all_covers_every_key(registry):
    keys = {entry["key"] for entry in registry.describe_all()}
    assert keys == {"alpha", "beta"}
