from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

from PyQt6.QtWidgets import QWidget

from pyrpoc.backend_utils.opto_control_contexts import BaseOptoControlContext
from pyrpoc.backend_utils.state_helpers import export_object_state, import_object_state, make_instance_id


class BaseOptoControlWidget(QWidget):
    """Base class for all optocontrol UI widgets.

    This lets the optocontrol manager mount a concrete editor widget without importing
    the control subtype. Concrete widgets should call `on_change` whenever user-editable
    values change so service/state handlers can persist updates.
    """

    def __init__(
        self,
        control: "BaseOptoControl",
        on_change: Callable[[], None] | None = None,
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self.control = control
        self.on_change = on_change

    def refresh_from_model(self) -> None:
        """Load current control data into widget fields.

        The widget manager calls this when a control is created from session restore.
        """
        return None

    def request_persist(self) -> None:
        if self.on_change is not None:
            self.on_change()


class BaseOptoControl(ABC):
    OPTOCONTROL_KEY: str = "base_optocontrol"
    DISPLAY_NAME: str = "Base Opto-Control"
    PERSISTENCE_FIELDS: tuple[str, ...] | None = None
    PERSISTENCE_EXCLUDE_FIELDS: tuple[str, ...] = (
        "alias",
        "connected",
        "enabled",
        "instance_id",
        "last_error",
        "user_label",
        "widget",
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
        self.alias = alias or self.OPTOCONTROL_KEY
        self.instance_id = instance_id or make_instance_id(self.alias)
        self.user_label = user_label
        self.enabled = enabled
        self.connected = bool(connected)
        self.last_error: str | None = None
        self.context: BaseOptoControlContext | None = None

    @property
    def type_key(self) -> str:
        '''Return the registry key used to recreate this control on restore.'''
        return self.alias

    @classmethod
    def get_contract(cls) -> dict[str, Any]:
        return {
            "optocontrol_key": cls.OPTOCONTROL_KEY,
            "display_name": cls.DISPLAY_NAME,
        }

    @abstractmethod
    def get_widget(
        self,
        parent: QWidget | None = None,
        on_change: Callable[[], None] | None = None,
        display_service: Any | None = None,
    ) -> BaseOptoControlWidget:
        """Create or reuse the editor widget for this specific control.

        Called by OptoControlManager UI handlers when it builds/refreshes the control list.
        """
        raise NotImplementedError

    def get_summary(self) -> str:
        """Return collapsed-row text for the control list view.

        The manager displays this in the compact control card and updates it after edits.
        """
        return self.user_label or self.DISPLAY_NAME

    def get_context(self) -> BaseOptoControlContext:
        """Build the control-specific acquisition context for the current run.

        Subclasses override this to pass modality-facing, acquisition-only state.
        """
        return BaseOptoControlContext(optocontrol_key=self.OPTOCONTROL_KEY, alias=self.alias)

    def cleanup(self) -> None:
        """Release widget/state resources before removal.

        Concrete controls override this when they hold threads, file handles, or external state.
        """
        self.last_error = None

    def prepare_for_acquisition(self) -> BaseOptoControlContext:
        """Default acquisition handoff for modality execution.

        This keeps `ModalityService` and existing callers simple by returning one
        context object per control rather than arbitrary payload tuples.
        """
        self.context = self.get_context()
        return self.context

    def export_persistence_state(self) -> dict[str, Any]:
        return export_object_state(
            self,
            include_fields=self.PERSISTENCE_FIELDS,
            exclude_fields=self.PERSISTENCE_EXCLUDE_FIELDS,
        )

    def import_persistence_state(self, state: dict[str, Any]) -> None:
        import_object_state(
            self,
            state,
            include_fields=self.PERSISTENCE_FIELDS,
            exclude_fields=self.PERSISTENCE_EXCLUDE_FIELDS,
        )
