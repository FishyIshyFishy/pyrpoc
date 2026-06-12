"""Drives the analysis overlays (ROIs, crosshair) and computes results.

The controller owns all interactive analysis state and writes results into the
:class:`AnalysisPanel`. It is deliberately agnostic about layout — it reaches
into the display only for the current data, channel names, and tiles.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pyqtgraph as pg

from .statistics import (
    ellipse_mask,
    empty_stats,
    line_profile,
    pixel_value,
    rectangle_mask,
    region_stats,
)
from .toolbar import AnalysisTool, tool_from_value

if TYPE_CHECKING:
    from .display import Tiled2DDisplay
    from .panel import AnalysisPanel
    from .tile import ChannelTile

roi_tools = (AnalysisTool.LINE, AnalysisTool.RECTANGLE, AnalysisTool.ELLIPSE)


class AnalysisController:
    def __init__(self, display: "Tiled2DDisplay", panel: "AnalysisPanel"):
        self._display = display
        self._panel = panel
        self._tool = AnalysisTool.PAN
        self._active_tile = 0
        self._roi: Any = None
        self._roi_tile = 0
        self._crosshairs: dict[int, tuple[Any, Any]] = {}
        self._pending_geometry: dict[str, Any] | None = None
        self._pending_tile: int | None = None

    def current_tool(self) -> AnalysisTool:
        return self._tool

    # ------------------------------------------------------------------
    # Tile lifecycle wiring
    # ------------------------------------------------------------------

    def register_tile(self, index: int, tile: "ChannelTile") -> None:
        scene: Any = tile.plot.scene()
        scene.sigMouseMoved.connect(lambda pos, i=index: self.on_mouse_moved(i, pos))

    def detach_tile(self, index: int) -> None:
        """Drop overlays bound to a tile that is about to be deleted."""
        if self._roi is not None and self._roi_tile == index:
            self.remove_roi()
        pair = self._crosshairs.pop(index, None)
        if pair is not None:
            tile = self._display.tile_at(index)
            if tile is not None:
                tile.plot.removeItem(pair[0])
                tile.plot.removeItem(pair[1])

    def tiles_synced(self) -> None:
        if self._active_tile >= self._display.tile_count():
            self._active_tile = 0

    # ------------------------------------------------------------------
    # Tool selection
    # ------------------------------------------------------------------

    def set_tool(
        self,
        tool: AnalysisTool,
        *,
        geometry: dict[str, Any] | None = None,
        roi_tile: int | None = None,
    ) -> None:
        self.remove_roi()
        self.remove_crosshairs()
        self._tool = tool
        if geometry is not None:
            self._pending_geometry = geometry
            self._pending_tile = roi_tile if roi_tile is not None else 0
        self._panel.show_tool(tool)
        if tool in roi_tools:
            self.create_roi(tool)
        self.refresh()

    def clear(self) -> None:
        self.remove_roi()
        self.remove_crosshairs()
        self._tool = AnalysisTool.PAN
        self._pending_geometry = None
        self._pending_tile = None
        self._panel.show_tool(AnalysisTool.PAN)

    def refresh(self) -> None:
        """Recompute the active tool's results against the current data."""
        if self._tool in roi_tools:
            if self._roi is None:
                self.create_roi(self._tool)
            if self._roi is None:
                return
        if self._tool in (AnalysisTool.RECTANGLE, AnalysisTool.ELLIPSE):
            self.compute_region_stats()
        elif self._tool == AnalysisTool.LINE:
            self.compute_line_profile()

    # ------------------------------------------------------------------
    # ROI management
    # ------------------------------------------------------------------

    def create_roi(self, tool: AnalysisTool) -> None:
        target = self._pending_tile if self._pending_tile is not None else self._active_tile
        tile = self._display.tile_at(target)
        if tile is None:
            return
        geometry = self._pending_geometry or self.default_geometry(tool)
        self._pending_geometry = None
        self._pending_tile = None

        pen = pg.mkPen(color=(0, 200, 255), width=2)
        if tool == AnalysisTool.LINE:
            roi = pg.LineSegmentROI(geometry["points"], pen=pen)
        elif tool == AnalysisTool.RECTANGLE:
            roi = pg.RectROI(geometry["pos"], geometry["size"], pen=pen)
        else:
            roi = pg.EllipseROI(geometry["pos"], geometry["size"], pen=pen)
            self.strip_rotate_handles(roi)
        roi.setZValue(20)
        tile.plot.addItem(roi)
        self._roi = roi
        self._roi_tile = target
        roi.sigRegionChanged.connect(self.on_roi_changed)
        roi.sigRegionChangeFinished.connect(self.on_roi_finished)

    def strip_rotate_handles(self, roi: Any) -> None:
        for info in list(roi.handles):
            if info.get("type") == "r":
                roi.removeHandle(info["item"])

    def remove_roi(self) -> None:
        if self._roi is None:
            return
        roi = self._roi
        self._roi = None
        try:
            roi.sigRegionChanged.disconnect(self.on_roi_changed)
            roi.sigRegionChangeFinished.disconnect(self.on_roi_finished)
        except (TypeError, RuntimeError):
            pass
        tile = self._display.tile_at(self._roi_tile)
        if tile is not None:
            tile.plot.removeItem(roi)

    def on_roi_changed(self) -> None:
        self.refresh()

    def on_roi_finished(self) -> None:
        self._display.request_persist()

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------

    def data_shape(self) -> tuple[int, int]:
        data = self._display.current_data()
        if data is not None:
            return int(data.shape[1]), int(data.shape[2])
        return 100, 100

    def default_geometry(self, tool: AnalysisTool) -> dict[str, Any]:
        height, width = self.data_shape()
        if tool == AnalysisTool.LINE:
            y = height / 2.0
            return {"points": [[width * 0.25, y], [width * 0.75, y]]}
        box_w = max(2.0, width / 3.0)
        box_h = max(2.0, height / 3.0)
        return {
            "pos": [width / 2.0 - box_w / 2.0, height / 2.0 - box_h / 2.0],
            "size": [box_w, box_h],
        }

    def roi_rect(self) -> tuple[float, float, float, float]:
        pos = self._roi.pos()
        size = self._roi.size()
        return float(pos[0]), float(pos[1]), float(size[0]), float(size[1])

    def line_endpoints(self) -> tuple[float, float, float, float]:
        points = [self._roi.mapToView(p) for p in self._roi.listPoints()]
        x0, y0 = float(points[0].x()), float(points[0].y())
        x1, y1 = float(points[1].x()), float(points[1].y())
        return x0, y0, x1, y1

    # ------------------------------------------------------------------
    # Computation
    # ------------------------------------------------------------------

    def compute_region_stats(self) -> None:
        data = self._display.current_data()
        names = self._display.get_channel_names()
        if data is None or self._roi is None:
            self._panel.show_stats(names, [empty_stats() for _ in names])
            return
        height, width = int(data.shape[1]), int(data.shape[2])
        x, y, w, h = self.roi_rect()
        if self._tool == AnalysisTool.RECTANGLE:
            mask = rectangle_mask(height, width, x, y, w, h)
        else:
            mask = ellipse_mask(height, width, x, y, w, h)
        stats = [region_stats(data[c], mask) for c in range(data.shape[0])]
        self._panel.show_stats(names, stats)

    def compute_line_profile(self) -> None:
        data = self._display.current_data()
        names = self._display.get_channel_names()
        if data is None or self._roi is None:
            self._panel.show_profiles(np.zeros(0, dtype=np.float64), names, [])
            return
        x0, y0, x1, y1 = self.line_endpoints()
        distances = np.zeros(0, dtype=np.float64)
        profiles: list[np.ndarray] = []
        for c in range(data.shape[0]):
            distances, values = line_profile(data[c], x0, y0, x1, y1)
            profiles.append(values)
        self._panel.show_profiles(distances, names, profiles)

    # ------------------------------------------------------------------
    # Pixel inspector
    # ------------------------------------------------------------------

    def on_mouse_moved(self, index: int, scene_pos: Any) -> None:
        self._active_tile = index
        if self._tool != AnalysisTool.INSPECT:
            return
        tile = self._display.tile_at(index)
        if tile is None:
            return
        view_point = tile.plot.getPlotItem().getViewBox().mapSceneToView(scene_pos)
        self.update_inspector(index, float(view_point.x()), float(view_point.y()))

    def update_inspector(self, index: int, x: float, y: float) -> None:
        data = self._display.current_data()
        names = self._display.get_channel_names()
        if data is None:
            self._panel.show_readout(None, None, names, [])
            return
        height, width = int(data.shape[1]), int(data.shape[2])
        col = int(np.floor(x))
        row = int(np.floor(y))
        if not (0 <= row < height and 0 <= col < width):
            self.hide_crosshairs()
            self._panel.show_readout(None, None, names, [])
            return
        self.ensure_crosshair(index)
        self.position_crosshair(index, col + 0.5, row + 0.5)
        values = [pixel_value(data[c], x, y) for c in range(data.shape[0])]
        self._panel.show_readout(col, row, names, values)

    def ensure_crosshair(self, index: int) -> None:
        self.hide_other_crosshairs(index)
        pair = self._crosshairs.get(index)
        if pair is not None:
            pair[0].show()
            pair[1].show()
            return
        tile = self._display.tile_at(index)
        if tile is None:
            return
        pen = pg.mkPen(color=(255, 220, 0, 160), width=1)
        vline = pg.InfiniteLine(angle=90, movable=False, pen=pen)
        hline = pg.InfiniteLine(angle=0, movable=False, pen=pen)
        vline.setZValue(15)
        hline.setZValue(15)
        tile.plot.addItem(vline, ignoreBounds=True)
        tile.plot.addItem(hline, ignoreBounds=True)
        self._crosshairs[index] = (vline, hline)

    def position_crosshair(self, index: int, x: float, y: float) -> None:
        pair = self._crosshairs.get(index)
        if pair is not None:
            pair[0].setPos(x)
            pair[1].setPos(y)

    def hide_other_crosshairs(self, keep: int) -> None:
        for idx, (vline, hline) in self._crosshairs.items():
            if idx != keep:
                vline.hide()
                hline.hide()

    def hide_crosshairs(self) -> None:
        for vline, hline in self._crosshairs.values():
            vline.hide()
            hline.hide()

    def remove_crosshairs(self) -> None:
        for idx, (vline, hline) in list(self._crosshairs.items()):
            tile = self._display.tile_at(idx)
            if tile is not None:
                tile.plot.removeItem(vline)
                tile.plot.removeItem(hline)
        self._crosshairs = {}

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def export_state(self) -> dict[str, Any]:
        state: dict[str, Any] = {"tool": self._tool.value, "active_tile": int(self._active_tile)}
        if self._roi is not None and self._tool in roi_tools:
            state["roi"] = self.roi_geometry()
            state["roi_tile"] = int(self._roi_tile)
        return state

    def roi_geometry(self) -> dict[str, Any]:
        if self._tool == AnalysisTool.LINE:
            x0, y0, x1, y1 = self.line_endpoints()
            return {"type": "line", "points": [[x0, y0], [x1, y1]]}
        x, y, w, h = self.roi_rect()
        return {"type": self._tool.value, "pos": [x, y], "size": [w, h]}

    def restore(self, state: dict[str, Any]) -> None:
        tool = tool_from_value(str(state.get("tool", "pan")))
        self._active_tile = int(state.get("active_tile", 0))
        geometry = None
        roi_tile = None
        roi = state.get("roi")
        if isinstance(roi, dict) and tool in roi_tools:
            geometry = self.geometry_from_state(roi)
            roi_tile = int(state.get("roi_tile", 0))
        self.set_tool(tool, geometry=geometry, roi_tile=roi_tile)

    def geometry_from_state(self, roi: dict[str, Any]) -> dict[str, Any]:
        if roi.get("type") == "line":
            points = roi.get("points", [[0.0, 0.0], [1.0, 1.0]])
            return {
                "points": [
                    [float(points[0][0]), float(points[0][1])],
                    [float(points[1][0]), float(points[1][1])],
                ]
            }
        pos = roi.get("pos", [0.0, 0.0])
        size = roi.get("size", [1.0, 1.0])
        return {"pos": [float(pos[0]), float(pos[1])], "size": [float(size[0]), float(size[1])]}
