"""Section 12's last invariant, checked at runtime.

"No Program subclass defines an attribute outside uses, params, emits, run."

The point is not tidiness. BaseModality grew emitted_kinds, allowed_displays,
get_frame_limit, prepare_acquisition_storage and the rest one caller at a time,
until its shape was set by whichever question was asked last. This fails the
moment that starts happening again.
"""

from __future__ import annotations

import pytest

import pyrpoc.programs  # noqa: F401  -- registers the programs
from pyrpoc.programs.registry import program_registry
from pyrpoc.run.program import Program

ALLOWED = {"uses", "params", "emits", "run"}


def declared(cls: type) -> set[str]:
    return {name for name in vars(cls) if not name.startswith("__")}


@pytest.mark.parametrize("key", program_registry.keys())
def test_program_declares_nothing_beyond_the_four_attributes(key):
    cls = program_registry.get(key)
    extra = declared(cls) - ALLOWED
    assert not extra, (
        f"{cls.__name__} declares {sorted(extra)}. A program's attributes are the "
        "three things the runner needs plus run(); anything else belongs in a "
        "module-level helper, shell/catalog.py, or the registry."
    )


@pytest.mark.parametrize("key", program_registry.keys())
def test_program_declares_all_four(key):
    cls = program_registry.get(key)
    assert declared(cls) == ALLOWED, f"{cls.__name__} is missing {sorted(ALLOWED - declared(cls))}"


@pytest.mark.parametrize("key", program_registry.keys())
def test_emits_maps_names_to_shape_contracts(key):
    from pyrpoc.core.streams import Stream

    cls = program_registry.get(key)
    assert cls.emits, f"{cls.__name__} emits nothing"
    for name, spec in cls.emits.items():
        assert isinstance(name, str) and name
        assert isinstance(spec, type) and issubclass(spec, Stream)


@pytest.mark.parametrize("key", program_registry.keys())
def test_uses_lists_device_classes(key):
    from pyrpoc.devices.base import Device

    cls = program_registry.get(key)
    for device_cls in cls.uses:
        assert isinstance(device_cls, type) and issubclass(device_cls, Device)


@pytest.mark.parametrize("key", program_registry.keys())
def test_params_is_a_dataclass_the_form_can_describe(key):
    from dataclasses import is_dataclass

    from pyrpoc.core import params as P

    cls = program_registry.get(key)
    assert is_dataclass(cls.params), f"{cls.__name__}.params must be a dataclass"
    assert P.sections(cls.params), f"{cls.__name__}.params describes no form sections"


def test_the_registry_does_not_stamp_a_key_onto_programs():
    """stamp=False, or registry_key would be a fifth attribute."""
    for key in program_registry.keys():
        assert "registry_key" not in vars(program_registry.get(key))


def test_the_base_class_supplies_no_loop():
    """v3.0's BaseModality owned `while not should_stop`. This one owns nothing."""
    assert not hasattr(Program, "acquire_continuous")
    assert not hasattr(Program, "build_continuous_worker")
    assert not hasattr(Program, "get_frame_limit")
    assert not hasattr(Program, "prepare_acquisition_storage")
    assert not hasattr(Program, "save_acquired_frame")
