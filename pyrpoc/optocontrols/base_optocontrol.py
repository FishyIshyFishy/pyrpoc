from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

from PyQt6.QtWidgets import QWidget


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


class BaseOptoControl(ABC):
    OPTOCONTROL_KEY: str = "base_optocontrol"
    DISPLAY_NAME: str = "Base Opto-Control"

    def __init__(self, alias: str | None = None, user_label: str | None = None, enabled: bool = False):
        self.alias = alias or self.OPTOCONTROL_KEY
        self.user_label = user_label
        self.enabled = enabled
        self.last_error: str | None = None

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

    @abstractmethod
    def prepare_for_acquisition(self) -> tuple[Any, ...]:
        """Build the control-specific acquisition payload for the current acquisition run.

        Called from optocontrol service when building the modality-specific run inputs.
        """
        raise NotImplementedError

    def cleanup(self) -> None:
        """Release widget/state resources before removal.

        Concrete controls override this when they hold threads, file handles, or external state.
        """
        self.last_error = None
