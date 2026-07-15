from __future__ import annotations

from typing import Any

from pyrpoc.acquisition.source import CommandSource
from pyrpoc.utils.state_helpers import export_object_state, import_object_state, make_instance_id
from pyrpoc.structs.contexts import BaseOptoControlContext


class BaseOptoControl:
    """Optocontrol logic (Qt-free).

    An optocontrol contributes to acquisition by supplying a source decorator
    via ``build_decorator``; the editor widget is provided separately by ``gui``.
    """

    optocontrol_key: str = "base_optocontrol"
    display_name: str = "Base Opto-Control"
    persistence_fields: tuple[str, ...] | None = None
    persistence_exclude_fields: tuple[str, ...] = (
        "alias",
        "connected",
        "enabled",
        "instance_id",
        "last_error",
        "user_label",
    )

    def __init__(
        self,
        alias: str | None = None,
        user_label: str | None = None,
        enabled: bool = False,
        *,
        instance_id: str | None = None,
        connected: bool = False,
    ):
        self.alias = alias or self.optocontrol_key
        self.instance_id = instance_id or make_instance_id(self.alias)
        self.user_label = user_label
        self.enabled = enabled
        self.connected = bool(connected)
        self.last_error: str | None = None
        self.context: BaseOptoControlContext | None = None

    @property
    def type_key(self) -> str:
        return self.alias

    @classmethod
    def get_contract(cls) -> dict[str, Any]:
        return {
            "optocontrol_key": cls.optocontrol_key,
            "display_name": cls.display_name,
        }

    def get_summary(self) -> str:
        return self.user_label or self.display_name

    def get_context(self) -> BaseOptoControlContext:
        """Build the control-specific acquisition context for the current run."""
        return BaseOptoControlContext(optocontrol_key=self.optocontrol_key, alias=self.alias)

    def prepare_for_acquisition(self) -> BaseOptoControlContext:
        self.context = self.get_context()
        return self.context

    def build_decorator(self, inner: CommandSource) -> CommandSource:
        """Wrap the command source to inject this control's behaviour.

        Default is a pass-through; concrete controls return a real decorator.
        """
        return inner

    def cleanup(self) -> None:
        self.last_error = None

    def export_persistence_state(self) -> dict[str, Any]:
        return export_object_state(
            self,
            include_fields=self.persistence_fields,
            exclude_fields=self.persistence_exclude_fields,
        )

    def import_persistence_state(self, state: dict[str, Any]) -> None:
        import_object_state(
            self,
            state,
            include_fields=self.persistence_fields,
            exclude_fields=self.persistence_exclude_fields,
        )
