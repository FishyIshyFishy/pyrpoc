"""A dialog for checking which entries of a grouped list to keep."""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

_KEY_ROLE = Qt.ItemDataRole.UserRole


class GroupedPickerDialog(QDialog):
    """Entries under their headings, each with a checkbox; ``checked`` holds the checked
    entries' data once accepted."""

    def __init__(
        self,
        title: str,
        groups: dict[str, list[tuple[str, str]]],
        checked: set[str],
        parent: QWidget,
    ):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(380, 520)
        root = QVBoxLayout(self)
        root.addLayout(self.build_search_row())
        self.tree = QTreeWidget(self)
        self.tree.setHeaderHidden(True)
        self.tree.setRootIsDecorated(False)
        self.entries: list[QTreeWidgetItem] = []
        self.fill(groups, checked)
        root.addWidget(self.tree, 1)
        root.addLayout(self.build_footer())

        self.search.textChanged.connect(self.apply_filter)
        self.tree.itemChanged.connect(lambda _item, _column: self.refresh_count())
        self.refresh_count()
        self.search.setFocus()

    def fill(self, groups: dict[str, list[tuple[str, str]]], checked: set[str]) -> None:
        """One tristate heading per group, its entries beneath."""
        for heading, entries in groups.items():
            group = QTreeWidgetItem(self.tree, [heading])
            group.setFlags(
                Qt.ItemFlag.ItemIsEnabled
                | Qt.ItemFlag.ItemIsUserCheckable
                | Qt.ItemFlag.ItemIsAutoTristate
            )
            font = group.font(0)
            font.setBold(True)
            group.setFont(0, font)
            for label, key in entries:
                item = QTreeWidgetItem(group, [label])
                item.setData(0, _KEY_ROLE, key)
                item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsUserCheckable)
                state = Qt.CheckState.Checked if key in checked else Qt.CheckState.Unchecked
                item.setCheckState(0, state)
                self.entries.append(item)
        self.tree.expandAll()

    def build_search_row(self) -> QHBoxLayout:
        """The search box, then All and None for what it shows."""
        row = QHBoxLayout()
        self.search = QLineEdit(self)
        self.search.setPlaceholderText("Search")
        self.search.setClearButtonEnabled(True)
        all_btn = QPushButton("All", self)
        all_btn.setToolTip("Check every entry shown")
        all_btn.setAutoDefault(False)
        all_btn.clicked.connect(lambda: self.check_shown(Qt.CheckState.Checked))
        none_btn = QPushButton("None", self)
        none_btn.setToolTip("Uncheck every entry shown")
        none_btn.setAutoDefault(False)
        none_btn.clicked.connect(lambda: self.check_shown(Qt.CheckState.Unchecked))
        row.addWidget(self.search, 1)
        row.addWidget(all_btn)
        row.addWidget(none_btn)
        return row

    def build_footer(self) -> QHBoxLayout:
        """The count, and OK and Cancel."""
        row = QHBoxLayout()
        self.count_label = QLabel(self)
        self.count_label.setStyleSheet("color: palette(mid);")
        self.buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, self
        )
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        row.addWidget(self.count_label, 1)
        row.addWidget(self.buttons)
        return row

    def apply_filter(self, text: str) -> None:
        """Show the entries whose label or heading contains ``text``."""
        query = text.strip().lower()
        for index in range(self.tree.topLevelItemCount()):
            group = self.tree.topLevelItem(index)
            assert group is not None
            heading_matches = query in group.text(0).lower()
            shown = 0
            for row in range(group.childCount()):
                item = group.child(row)
                assert item is not None
                hidden = not heading_matches and query not in item.text(0).lower()
                item.setHidden(hidden)
                shown += not hidden
            group.setHidden(shown == 0)

    def check_shown(self, state: Qt.CheckState) -> None:
        for item in self.entries:
            if not item.isHidden():
                item.setCheckState(0, state)

    def refresh_count(self) -> None:
        """Say how many are checked; OK needs at least one."""
        count = len(self.checked)
        self.count_label.setText(f"{count} of {len(self.entries)}")
        ok = self.buttons.button(QDialogButtonBox.StandardButton.Ok)
        assert ok is not None
        ok.setEnabled(count > 0)

    @property
    def checked(self) -> set[str]:
        return {
            item.data(0, _KEY_ROLE)
            for item in self.entries
            if item.checkState(0) == Qt.CheckState.Checked
        }
