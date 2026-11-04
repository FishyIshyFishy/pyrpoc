"""Shorthands that build a common rule and declare a program input or a device setting with it."""

from __future__ import annotations

from pyrpoc.core.device_config import spec_field
from pyrpoc.core.program_input import ProgramInput, declare

from .rules import BoolField, ChannelsField, ChoiceField, FloatField, IntField, PathField, TextField


def int_input(
    key, label, default, *, minimum=None, maximum=None, step=1, tooltip=""
) -> ProgramInput[int]:
    return declare(key, IntField(label, tooltip, minimum, maximum, step), default)


def float_input(
    key, label, default, *, minimum=None, maximum=None, step=0.1, decimals=6, tooltip=""
) -> ProgramInput[float]:
    return declare(key, FloatField(label, tooltip, minimum, maximum, step, decimals), default)


def bool_input(key, label, default, *, tooltip="") -> ProgramInput[bool]:
    return declare(key, BoolField(label, tooltip), default)


def choice_input(key, label, default, *, choices, tooltip="") -> ProgramInput[str]:
    return declare(key, ChoiceField(label, tooltip, tuple(choices)), default)


def int_field(label, default, *, minimum=None, maximum=None, step=1, tooltip=""):
    return spec_field(default, IntField(label, tooltip, minimum, maximum, step))


def float_field(label, default, *, minimum=None, maximum=None, step=0.1, decimals=6, tooltip=""):
    return spec_field(default, FloatField(label, tooltip, minimum, maximum, step, decimals))


def text_field(label, default="", *, tooltip=""):
    return spec_field(default, TextField(label, tooltip))


def path_field(label, default="", *, dialog_filter="All Files (*)", tooltip=""):
    return spec_field(default, PathField(label, tooltip, dialog_filter))


def channels_field(label, *, num_channels=9, default=None, tooltip=""):
    resolved = tuple(range(num_channels)) if default is None else tuple(default)
    return spec_field(None, ChannelsField(label, tooltip, num_channels), factory=lambda: resolved)
