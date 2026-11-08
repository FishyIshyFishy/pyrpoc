"""``SelectedCard``: the routine chosen in the queue: where it stands, its last scan, its unit's
latest frame, and every measurement it recorded, one small trace per channel."""

from __future__ import annotations

from collections.abc import Sequence

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QFont, QPixmap
from PyQt6.QtWidgets import QLabel, QVBoxLayout, QWidget

from pyrpoc.plugins.program_io.outputs import Series, UnitScanned, UnitView
from pyrpoc.utils.widgets.colors import color_for_index

from .. import look
from ..feed import LoopFeed, UnitFrame
from ..render import levels_of, pixmap_of, rgb_of
from ..state.history import last_scan
from ..state.narration import duration, row_text
from ..state.routines import RoutineKey
from ..state.snapshot import Snapshot
from .card import Anchor, Card
from .channel_trace import ChannelTrace

FRAME_PX = 240


def scan_text(scan: UnitScanned | None, now_s: float) -> str:
    if scan is None:
        return "not scanned yet"
    found = "found" if scan.found else "<b>lost</b>: not found where it was"
    moved = (scan.shift_um[0] ** 2 + scan.shift_um[1] ** 2) ** 0.5
    late = f" · ran {duration(scan.late_s)} late" if scan.late_s >= 1 else ""
    return f"{duration(now_s - scan.time_s)} ago · {found} · moved {moved:.1f} µm{late}"


def fact(label: str, text: str) -> str:
    return f"<span style='color:{look.TEXT_DIM.name()}'>{label}</span>&nbsp; {text}"


class SelectedCard(Card):
    def __init__(self, parent: QWidget, feed: LoopFeed, width: int):
        super().__init__(parent, "Chosen routine", Anchor.RIGHT, width, 3)
        self.feed = feed
        self.chosen: RoutineKey | None = None
        self.framed: tuple[int, int] | None = None
        self.traced: tuple[tuple[str, ...], tuple[str, ...]] | None = None
        self.traces: list[list[ChannelTrace]] = []
        self.title = self.label(look.font(11.0, True))
        self.facts = self.label(look.font(8.5, False))
        self.frame = QLabel(self)
        self.frame.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.caption = self.label(look.font(7.5, False))
        self.measures = QVBoxLayout()
        self.measures.setSpacing(2)
        for widget in (self.title, self.facts, self.frame, self.caption):
            self.content.addWidget(widget)
        self.content.addLayout(self.measures)
        self.content.addStretch(1)

    def label(self, face: QFont) -> QLabel:
        label = QLabel("", self)
        label.setFont(face)
        label.setWordWrap(True)
        label.setTextFormat(Qt.TextFormat.RichText)
        label.setStyleSheet(f"color: {look.TEXT.name()};")
        return label

    def choose(self, key: RoutineKey | None) -> None:
        self.chosen = key

    def show_snapshot(self, snapshot: Snapshot) -> None:
        key = self.chosen
        unit = None if key is None else snapshot.unit(key[0])
        if key is None or unit is None:
            self.show_nothing(key)
            return
        self.show_standing(snapshot, key, unit)
        self.show_frame(self.feed.frames().get(unit.number), snapshot.now_s)
        series = [s for s in unit.series if s.routine == key[1]]
        self.show_measures(series, snapshot.view.channels)

    def show_nothing(self, key: RoutineKey | None) -> None:
        if key is None:
            self.title.setText("")
            self.facts.setText("Choose a routine in the queue to see it here.")
        else:
            self.title.setText(f"Unit {key[0]} has left the loop")
            self.facts.setText("")
        self.frame.clear()
        self.caption.setText("")
        self.framed = None
        self.show_measures([], ())

    def show_standing(self, snapshot: Snapshot, key: RoutineKey, unit: UnitView) -> None:
        dot = look.STATE[unit.state].name()
        group = f" · {unit.group}" if unit.group else ""
        self.title.setText(
            f"<span style='color:{dot}'>●</span> Unit {unit.number} · {key[1]}"
            f"<span style='color:{look.TEXT_DIM.name()}'>&nbsp; {unit.state.value}{group}</span>"
        )
        row = next((r for r in snapshot.rows if r.key == key), None)
        standing = "ended" if row is None else row_text(row.section, row.status, snapshot)
        scan = last_scan(snapshot.events, key)
        frames = 0 if scan is None else scan.frame + 1
        self.facts.setText(
            "<br>".join(
                (
                    fact("Next", standing),
                    fact("Last scan", scan_text(scan, snapshot.now_s)),
                    fact("Frames", str(frames)),
                )
            )
        )

    def show_frame(self, frame: UnitFrame | None, now_s: float) -> None:
        """The unit's newest frame, drawn again only when a new one comes."""
        if frame is None:
            self.frame.clear()
            self.caption.setText("no frame yet")
            self.framed = None
            return
        shown = (frame.tag.unit, frame.serial)
        if shown != self.framed:
            self.framed = shown
            self.frame.setPixmap(self.thumbnail(frame))
        self.caption.setText(
            f"<span style='color:{look.TEXT_DIM.name()}'>latest frame · {frame.tag.routine}</span>"
        )

    def thumbnail(self, frame: UnitFrame) -> QPixmap:
        planes = frame.planes
        colours = [color_for_index(index) for index in range(planes.shape[0])]
        pixmap = pixmap_of(rgb_of(planes, levels_of(planes), colours))
        side = min(FRAME_PX, self.width() - 28)
        return pixmap.scaled(
            side,
            side,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )

    def show_measures(self, series: Sequence[Series], channels: tuple[str, ...]) -> None:
        """One heading per measurement, then one trace per channel, since channels differ in
        scale and so never share an axis."""
        names = tuple(s.name for s in series)
        if (names, channels) != self.traced:
            self.traced = (names, channels)
            self.rebuild(names, channels)
        for s, traces in zip(series, self.traces, strict=True):
            for index, trace in enumerate(traces):
                trace.show_points([(t, values[index]) for t, values in s.points])

    def rebuild(self, names: tuple[str, ...], channels: tuple[str, ...]) -> None:
        while self.measures.count():
            item = self.measures.takeAt(0)
            widget = None if item is None else item.widget()
            if widget is not None:
                widget.deleteLater()
        self.traces = []
        for name in names:
            heading = self.label(look.font(7.5, True))
            heading.setText(f"<span style='color:{look.TEXT_DIM.name()}'>{name.upper()}</span>")
            self.measures.addWidget(heading)
            traces = [
                ChannelTrace(self, channel, QColor(*color_for_index(index)))
                for index, channel in enumerate(channels)
            ]
            for trace in traces:
                self.measures.addWidget(trace)
            self.traces.append(traces)
