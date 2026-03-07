from __future__ import annotations

from PyQt6.QtWidgets import QLabel, QVBoxLayout, QWidget

from .base_optocontrol import BaseOptoControl, BaseOptoControlWidget
from .opto_control_registry import opto_control_registry


class MaskOptoControlWidget(BaseOptoControlWidget):
    """Placeholder control widget for mask configuration.

    This class is intentionally light so the control implementation can evolve
    without changing the `OptoControlManager` container contract.
    """

    def __init__(self, control: "MaskOptoControl", parent: QWidget | None = None) -> None:
        super().__init__(control, parent=parent)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Mask opto-control UI placeholder", self))
        layout.addWidget(QLabel("Implement this control's full widget later.", self))
        layout.addStretch(1)

    def refresh_from_model(self) -> None:
        pass


@opto_control_registry.register("mask")
class MaskOptoControl(BaseOptoControl):
    OPTOCONTROL_KEY = "mask"
    DISPLAY_NAME = "Mask Opto-Control"

    def __init__(
        self,
        alias: str | None = None,
        user_label: str | None = None,
        enabled: bool = False,
    ):
        super().__init__(alias=alias or self.OPTOCONTROL_KEY, user_label=user_label, enabled=enabled)
        self.widget: QWidget | None = None

    def get_widget(
        self,
        parent: QWidget | None = None,
        on_change=None,
    ) -> BaseOptoControlWidget:
        """Create the placeholder widget used when this control card is expanded.

        Called from:
        - `optocontrol_service.get_widget`
        - `opto_control_mgr.handlers.on_expand_requested`

        Returns:
            Reused widget instance for this control.
        """
        if self.widget is None:
            self.widget = MaskOptoControlWidget(self, parent=parent)
        elif parent is not None and self.widget.parent() is not None:
            self.widget.setParent(parent)
        if on_change is not None and isinstance(self.widget, BaseOptoControlWidget):
            self.widget.on_change = on_change
        return self.widget

    def prepare_for_acquisition(self) -> tuple[str, dict[str, str]]:
        """Build the control's acquisition tuple.

        Called from `OptoControlService.collect_data_for_acquisition` and forwarded to
        modality execution. The tuple shape is stable for this subtype.
        """
        return (self.type_key, {"alias": self.alias})


Mask = MaskOptoControl
