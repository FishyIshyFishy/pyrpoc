"""Shared card widgets used by all manager panels.

Hierarchy
---------
BaseCardWidget
    Core collapsible card: expand arrow, optional toggle checkbox, title label,
    collapsible body area, and a muted description line shown while collapsed.

RemovableCardWidget(BaseCardWidget)
    Adds a pink "X" remove button to the header row.  Used by instrument,
    opto-control, and display managers.  Acquisition parameter-group cards use
    BaseCardWidget directly because they are never individually removable.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QCheckBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

# Accent used for the remove "X" button and for cards that want a fixed pink
# accent instead of the palette highlight.  The toggle checkbox in
# BaseCardWidget always uses the palette highlight colour so it integrates
# naturally with whatever theme is loaded.
_REMOVE_BTN_COLOR = "#e75480"


class BaseCardWidget(QFrame):
    """Collapsible card with an optional toggle checkbox.

    Signals
    -------
    expand_requested(object)
        Emitted when the expand/collapse arrow is clicked.  Passes ``state_obj``
        so the owning manager can do lazy widget construction.
    toggle_changed(object, bool)
        Emitted when the checkbox state changes.  Passes ``state_obj`` and the
        new checked value.  Connect this to enable/disable or attach/detach
        handlers as appropriate.
    """

    expand_requested = pyqtSignal(object)
    toggle_changed = pyqtSignal(object, bool)

    def __init__(
        self,
        state_obj: object,
        title: str,
        parent: QWidget | None = None,
        *,
        card_name: str = "instanceCard",
        accent_color: str | None = None,
    ) -> None:
        """
        Parameters
        ----------
        state_obj:
            Arbitrary object used as an identity token for signals and manager
            card maps.  Passed back verbatim in all signals.
        title:
            Initial text shown in the card header label.
        card_name:
            QSS object-name used for stylesheet scoping.  Override when you
            need separate stylesheet rules for a specific card type.
        accent_color:
            Hex string (e.g. ``"#e75480"``) used for the checkbox checked
            background and the active-card border/background tint.  Defaults to
            ``None``, which uses ``palette(highlight)``.
        """
        super().__init__(parent)
        self.state_obj = state_obj
        self._expanded = False
        self._toggle_guard = False
        self._card_name = card_name
        self._accent_color = accent_color

        self.setObjectName(card_name)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setStyleSheet(self._compose_stylesheet(active=False))

        root = QVBoxLayout(self)
        root.setContentsMargins(6, 4, 6, 4)
        root.setSpacing(2)

        # ── Header row ──────────────────────────────────────────────────────
        self._header_row = QHBoxLayout()
        self._header_row.setContentsMargins(0, 0, 0, 0)
        self._header_row.setSpacing(4)

        self.expand_btn = QToolButton(self)
        self.expand_btn.setArrowType(Qt.ArrowType.RightArrow)
        self.expand_btn.setAutoRaise(True)
        self.expand_btn.setToolTip("Expand")
        self.expand_btn.clicked.connect(self._on_expand_clicked)
        self._header_row.addWidget(self.expand_btn)

        self.toggle_checkbox = QCheckBox("", self)
        self.toggle_checkbox.setToolTip("Toggle")
        self.toggle_checkbox.toggled.connect(self._on_toggle_changed)
        self._header_row.addWidget(self.toggle_checkbox)

        self.title_label = QLabel(title, self)
        self._header_row.addWidget(self.title_label, 1)

        # Subclasses may insert additional header widgets here via
        # _append_header_widget() before the layout is finalised.
        self._finalise_header(root)

        # ── Description line (shown collapsed only) ──────────────────────────
        self._description_label = QLabel("", self)
        self._description_label.setStyleSheet(
            "color: palette(mid); font-size: 9pt; padding-left: 22px;"
        )
        self._description_label.setWordWrap(True)
        self._description_label.setVisible(False)
        root.addWidget(self._description_label)

        # ── Body (lazy-populated by owning manager) ───────────────────────────
        self.body_container = QWidget(self)
        self.body_layout = QVBoxLayout(self.body_container)
        self.body_layout.setContentsMargins(0, 2, 0, 0)
        self.body_layout.setSpacing(4)
        self.body_container.setVisible(False)
        root.addWidget(self.body_container)

    # ------------------------------------------------------------------
    # Subclass extension points
    # ------------------------------------------------------------------

    def _finalise_header(self, root: QVBoxLayout) -> None:
        """Called at the end of __init__ to add the header row to *root*.

        Subclasses that need to insert extra widgets into ``_header_row``
        before it is committed should override this, call super() last.
        """
        root.addLayout(self._header_row)

    # ------------------------------------------------------------------
    # Internal slots
    # ------------------------------------------------------------------

    def _on_expand_clicked(self) -> None:
        self.expand_requested.emit(self.state_obj)

    def _on_toggle_changed(self, checked: bool) -> None:
        self._apply_active_visual(bool(checked))
        if self._toggle_guard:
            return
        self.toggle_changed.emit(self.state_obj, checked)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_expanded(self, expanded: bool) -> None:
        self._expanded = expanded
        self.body_container.setVisible(expanded)
        self.expand_btn.setArrowType(
            Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow
        )
        self.expand_btn.setToolTip("Collapse" if expanded else "Expand")
        if self._description_label.text():
            self._description_label.setVisible(not expanded)

    def is_expanded(self) -> bool:
        return self._expanded

    def set_toggle_checked(self, checked: bool, guarded: bool = True) -> None:
        if guarded:
            self._toggle_guard = True
        self.toggle_checkbox.setChecked(checked)
        self._apply_active_visual(bool(checked))
        if guarded:
            self._toggle_guard = False

    def set_toggle_visible(self, visible: bool) -> None:
        self.toggle_checkbox.setVisible(visible)

    def set_description(self, text: str) -> None:
        self._description_label.setText(text)
        self._description_label.setStyleSheet("color: white;")
        self._description_label.setVisible(bool(text) and not self._expanded)

    def set_marker_text(self, text: str) -> None:  # noqa: ARG002
        """No-op kept for call-site compatibility.  Override if needed."""

    def set_local_status(self, text: str) -> None:
        self.setToolTip(text)

    def set_body_widget(self, body: QWidget | None) -> None:
        while self.body_layout.count():
            item = self.body_layout.takeAt(0)
            child = item.widget()
            if child is not None:
                child.setParent(None)  # type: ignore[arg-type]
                child.deleteLater()
        if body is not None:
            self.body_layout.addWidget(body)

    # ------------------------------------------------------------------
    # Backward-compat aliases (used by existing opto/instrument callers)
    # ------------------------------------------------------------------

    def set_enable_checked(self, checked: bool, guarded: bool = True) -> None:
        self.set_toggle_checked(checked, guarded)

    def set_enable_visible(self, visible: bool) -> None:
        self.set_toggle_visible(visible)

    # ------------------------------------------------------------------
    # Visuals
    # ------------------------------------------------------------------

    def _apply_active_visual(self, active: bool) -> None:
        self.setStyleSheet(self._compose_stylesheet(active=bool(active)))
        self.style().unpolish(self)
        self.style().polish(self)
        self.update()

    def _accent_css_checked(self) -> str:
        """Return the CSS value used for the checked-checkbox background."""
        if self._accent_color:
            return self._accent_color
        return "palette(highlight)"

    def _accent_css_active_bg(self) -> str:
        if self._accent_color:
            # derive a very muted tint from the hex colour
            r, g, b = _hex_to_rgb(self._accent_color)
            return f"rgba({r}, {g}, {b}, 18)"
        # palette-based tint
        highlight = QColor(self.palette().color(self.palette().ColorRole.Highlight))
        muted = QColor(highlight)
        muted.setAlpha(46)
        return f"rgba({muted.red()}, {muted.green()}, {muted.blue()}, {muted.alpha()})"

    def _accent_css_active_border(self) -> str:
        if self._accent_color:
            r, g, b = _hex_to_rgb(self._accent_color)
            return f"rgba({r}, {g}, {b}, 150)"
        highlight = QColor(self.palette().color(self.palette().ColorRole.Highlight))
        muted = QColor(highlight)
        muted.setAlpha(150)
        return f"rgba({muted.red()}, {muted.green()}, {muted.blue()}, {muted.alpha()})"

    def _compose_stylesheet(self, active: bool) -> str:
        n = self._card_name
        checked_bg = self._accent_css_checked()
        active_block = ""
        if active:
            active_block = (
                f"#{n} {{"
                f"background: {self._accent_css_active_bg()};"
                f"border: 1px solid {self._accent_css_active_border()};"
                "}"
            )
        return (
            f"#{n} {{"
            "background: palette(base);"
            "border: 1px solid palette(midlight);"
            "border-radius: 6px;"
            "}"
            f"#{n} > QWidget, #{n} > QWidget QWidget {{"
            "background: transparent;"
            "}"
            f"#{n} QLabel, #{n} QCheckBox, #{n} QToolButton {{"
            "background: transparent;"
            "}"
            f"#{n} QCheckBox::indicator:checked {{"
            f"background: {checked_bg};"
            f"border: 1px solid {checked_bg};"
            "color: white;"
            "}"
            f"#{n} QCheckBox::indicator:unchecked {{"
            "background: transparent;"
            "border: 1px solid palette(mid);"
            "}"
            f"{active_block}"
        )


class RemovableCardWidget(BaseCardWidget):
    """BaseCardWidget with a pink "X" remove button in the header.

    Used by instrument, opto-control, and display managers.

    Additional signal
    -----------------
    remove_requested(object)
        Emitted when the X button is clicked.  Passes ``state_obj``.
    """

    remove_requested = pyqtSignal(object)

    def __init__(
        self,
        state_obj: object,
        title: str,
        parent: QWidget | None = None,
        *,
        card_name: str = "instanceCard",
        accent_color: str | None = None,
    ) -> None:
        # remove_btn must exist before _finalise_header is called by super().__init__
        self.remove_btn = QToolButton()
        self.remove_btn.setAutoRaise(True)
        self.remove_btn.setText("X")
        self.remove_btn.setStyleSheet(
            f"QToolButton {{ color: {_REMOVE_BTN_COLOR}; font-weight: 700; }}"
        )
        self.remove_btn.setToolTip("Remove")
        super().__init__(
            state_obj, title, parent, card_name=card_name, accent_color=accent_color
        )
        # Wire signal after super().__init__ so self is fully constructed
        self.remove_btn.clicked.connect(
            lambda: self.remove_requested.emit(self.state_obj)
        )

    def _finalise_header(self, root: QVBoxLayout) -> None:
        """Insert remove button before committing header row to layout."""
        self.remove_btn.setParent(self)  # type: ignore[arg-type]
        self._header_row.addWidget(self.remove_btn)
        root.addLayout(self._header_row)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    """Parse ``"#rrggbb"`` → ``(r, g, b)``."""
    h = hex_color.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
