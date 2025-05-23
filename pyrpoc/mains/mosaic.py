from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QGroupBox, QGridLayout, QLabel, QSpinBox,
    QComboBox, QPushButton, QScrollArea, QWidget, QCheckBox, QFileDialog,
    QHBoxLayout, QLineEdit, QDoubleSpinBox
)
from PyQt5.QtGui import QPixmap, QImage, QPainter, QColor, QPen
from PyQt5.QtCore import Qt, QTimer, QRunnable, QThread, QThreadPool, QObject, pyqtSignal, pyqtSlot
from pyrpoc.mains import acquisition
from pyrpoc.helpers.prior_stage.functions import *
import numpy as np
import random
import json
import os
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd

class MosaicCanvas:
    def __init__(self, tile_w, tile_h, step_px, rows, cols):
        self.tile_w = tile_w
        self.tile_h = tile_h
        self.step_px = step_px
        self.rows = rows
        self.cols = cols

        self.canvas_w = step_px * (cols - 1) + tile_w
        self.canvas_h = step_px * (rows - 1) + tile_h

        self.canvas_rgb = np.zeros((self.canvas_h, self.canvas_w, 3), dtype=np.float32)
        self.weight_map = np.zeros((self.canvas_h, self.canvas_w), dtype=np.float32)

        yy, xx = np.meshgrid(np.linspace(-1, 1, tile_h), np.linspace(-1, 1, tile_w), indexing='ij')
        ramp = 0.5 * (1 + np.cos(np.pi * np.clip(np.maximum(np.abs(xx), np.abs(yy)), 0, 1)))
        self.ramp = ramp.astype(np.float32)

    def blend_tile(self, i, j, tile_rgb):
        x = j * self.step_px
        y = i * self.step_px
        h, w = self.tile_h, self.tile_w

        ramp = self.ramp[..., None]
        self.canvas_rgb[y:y+h, x:x+w, :] += tile_rgb * ramp
        self.weight_map[y:y+h, x:x+w] += self.ramp

    def render_qimage(self):
        norm_weight = np.clip(self.weight_map, 1e-6, None)[..., None]
        blended = self.canvas_rgb / norm_weight
        rgb8 = (np.clip(blended, 0, 1) * 255).astype(np.uint8)
        h, w = rgb8.shape[:2]
        return QImage(rgb8.data, w, h, 3 * w, QImage.Format_RGB888).copy()

    def save_to_file(self, path):
        norm_weight = np.clip(self.weight_map, 1e-6, None)[..., None]
        blended = self.canvas_rgb / norm_weight
        img = (np.clip(blended, 0, 1) * 255).astype(np.uint8)
        Image.fromarray(img).save(path)



class ZoomableLabel(QLabel):
    def __init__(self, scroll_area=None):
        super().__init__()
        self.scroll_area = scroll_area
        self._pixmap = None
        self.scale_factor = 1.0
        self._drag_pos = None
        self.setMouseTracking(True)
        self.setAlignment(Qt.AlignLeft | Qt.AlignTop)

    def setPixmap(self, pixmap: QPixmap):
        self._pixmap = pixmap
        self.repaint_scaled()

    def wheelEvent(self, event):
        if not self._pixmap:
            return

        old_pos = event.pos()
        old_scroll_x = self.scroll_area.horizontalScrollBar().value()
        old_scroll_y = self.scroll_area.verticalScrollBar().value()

        offset_x = old_pos.x() + old_scroll_x
        offset_y = old_pos.y() + old_scroll_y

        angle = event.angleDelta().y()
        factor = 1.25 if angle > 0 else 0.8
        new_scale = self.scale_factor * factor
        new_scale = max(0.1, min(new_scale, 20))

        if new_scale == self.scale_factor:
            return

        self.scale_factor = new_scale
        self.repaint_scaled()

        new_scroll_x = int(offset_x * factor - old_pos.x())
        new_scroll_y = int(offset_y * factor - old_pos.y())

        self.scroll_area.horizontalScrollBar().setValue(new_scroll_x)
        self.scroll_area.verticalScrollBar().setValue(new_scroll_y)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._drag_pos = event.pos()

    def mouseMoveEvent(self, event):
        if self._drag_pos and self.scroll_area:
            diff = event.pos() - self._drag_pos
            self.scroll_area.horizontalScrollBar().setValue(
                self.scroll_area.horizontalScrollBar().value() - diff.x())
            self.scroll_area.verticalScrollBar().setValue(
                self.scroll_area.verticalScrollBar().value() - diff.y())
            self._drag_pos = event.pos()

    def mouseReleaseEvent(self, event):
        self._drag_pos = None

    def repaint_scaled(self):
        if self._pixmap:
            scaled = self._pixmap.scaled(
                self._pixmap.size() * self.scale_factor,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            super().setPixmap(scaled)

class MosaicCanvas:
    def __init__(self, tile_w, tile_h, step_px, rows, cols):
        self.tile_w = tile_w
        self.tile_h = tile_h
        self.step_px = step_px
        self.rows = rows
        self.cols = cols

        self.canvas_w = step_px * (cols - 1) + tile_w
        self.canvas_h = step_px * (rows - 1) + tile_h

        self.canvas_rgb = np.zeros((self.canvas_h, self.canvas_w, 3), dtype=np.float32)
        self.weight_map = np.zeros((self.canvas_h, self.canvas_w), dtype=np.float32)

        # Precomputed weight mask (cosine ramp)
        yy, xx = np.meshgrid(np.linspace(-1, 1, tile_h), np.linspace(-1, 1, tile_w), indexing='ij')
        ramp = 0.5 * (1 + np.cos(np.pi * np.clip(np.maximum(np.abs(xx), np.abs(yy)), 0, 1)))
        self.ramp = ramp.astype(np.float32)

    def blend_tile(self, i, j, tile_rgb):
        x = j * self.step_px
        y = i * self.step_px
        h, w = self.tile_h, self.tile_w

        ramp = self.ramp[..., None]
        self.canvas_rgb[y:y+h, x:x+w, :] += tile_rgb * ramp
        self.weight_map[y:y+h, x:x+w] += self.ramp

    def render_qimage(self):
        norm_weight = np.clip(self.weight_map, 1e-6, None)[..., None]
        blended = self.canvas_rgb / norm_weight
        rgb8 = (np.clip(blended, 0, 1) * 255).astype(np.uint8)
        h, w = rgb8.shape[:2]
        return QImage(rgb8.data, w, h, 3 * w, QImage.Format_RGB888).copy()


class MosaicDialog(QDialog):
    def __init__(self, main_gui, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Mosaic Imaging")
        self.main_gui = main_gui
        self.cancelled = False
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.setFocusPolicy(Qt.StrongFocus)
        self.resize(1200, 800)

        main_layout = QHBoxLayout(self)
        self.setLayout(main_layout)

        self.sidebar_widget = QWidget()
        self.sidebar_layout = QVBoxLayout(self.sidebar_widget)
        self.sidebar_layout.setContentsMargins(10, 10, 10, 10)
        self.sidebar_layout.setSpacing(12)
        self.sidebar_widget.setFixedWidth(300)

        self.save_group = QGroupBox("Save Options")
        save_layout = QGridLayout(self.save_group)

        save_layout.addWidget(QLabel("Save Folder:"), 0, 0)
        self.save_folder_entry = QLineEdit()
        self.save_folder_entry.setPlaceholderText("Select folder...")
        save_layout.addWidget(self.save_folder_entry, 0, 1)
        browse_btn = QPushButton("📂")
        browse_btn.clicked.connect(self.browse_save_folder)
        save_layout.addWidget(browse_btn, 0, 2)

        self.save_metadata_checkbox = QCheckBox("Save Metadata (.json)")
        self.save_stitched_checkbox = QCheckBox("Save Stitched TIFF")
        self.save_tiles_checkbox = QCheckBox("Save Individual Tiles")
        self.save_averages_checkbox = QCheckBox("Save Averages of Repetitions")
        self.save_focus_metrics_checkbox = QCheckBox("Save Tile Focus Metrics")

        self.save_metadata_checkbox.setChecked(False)
        self.save_stitched_checkbox.setChecked(False)
        self.save_tiles_checkbox.setChecked(False)
        self.save_averages_checkbox.setChecked(False)
        self.save_focus_metrics_checkbox.setChecked(False)

        save_layout.addWidget(self.save_metadata_checkbox, 1, 0, 1, 3)
        save_layout.addWidget(self.save_stitched_checkbox, 2, 0, 1, 3)
        save_layout.addWidget(self.save_tiles_checkbox, 3, 0, 1, 3)
        save_layout.addWidget(self.save_averages_checkbox, 4, 0, 1, 3)
        save_layout.addWidget(self.save_focus_metrics_checkbox, 5, 0, 1, 3)

        params_group = QGroupBox("Mosaic Parameters")
        params_layout = QGridLayout(params_group)

        self.rows_spin = QSpinBox(); self.rows_spin.setRange(1, 1000); self.rows_spin.setValue(3)
        self.cols_spin = QSpinBox(); self.cols_spin.setRange(1, 1000); self.cols_spin.setValue(3)
        self.overlap_spin = QSpinBox(); self.overlap_spin.setRange(0, 100); self.overlap_spin.setValue(10)
        self.repetitions_spin = QSpinBox(); self.repetitions_spin.setRange(1, 100); self.repetitions_spin.setValue(1)
        self.pattern_combo = QComboBox(); self.pattern_combo.addItems(["Snake", "Raster"])
        self.fov_um_spin = QSpinBox(); self.fov_um_spin.setRange(1, 10000); self.fov_um_spin.setValue(100)
        self.repetitions_spin = QSpinBox(); self.repetitions_spin.setRange(1,100); self.repetitions_spin.setValue(1)
        self.grid_checkbox = QCheckBox("Show Tile Grid"); self.grid_checkbox.setChecked(True)
        self.display_mosaic_checkbox = QCheckBox("Display Mosaic Live")
        self.display_mosaic_checkbox.setChecked(True)
        
        
        self.grid_checkbox.stateChanged.connect(self.update_display)
        self.rows_spin.valueChanged.connect(self.report_memory_estimate)
        self.cols_spin.valueChanged.connect(self.report_memory_estimate)

        
        params_layout.addWidget(QLabel("Rows:"), 0, 0)
        params_layout.addWidget(self.rows_spin, 0, 1)
        params_layout.addWidget(QLabel("Columns:"), 1, 0)
        params_layout.addWidget(self.cols_spin, 1, 1)
        params_layout.addWidget(QLabel("Overlap (%):"), 2, 0)
        params_layout.addWidget(self.overlap_spin, 2, 1)
        params_layout.addWidget(QLabel('Repetitions per Tile'), 3, 0)
        params_layout.addWidget(self.repetitions_spin, 3, 1)
        params_layout.addWidget(QLabel("Pattern:"), 4, 0)
        params_layout.addWidget(self.pattern_combo, 4, 1)
        params_layout.addWidget(QLabel("FOV Size (μm):"), 5, 0)
        params_layout.addWidget(self.fov_um_spin, 5, 1)
        params_layout.addWidget(self.grid_checkbox, 6, 0, 1, 2)
        params_layout.addWidget(self.display_mosaic_checkbox, 7, 0, 1, 2)

        self.start_button = QPushButton("Start Mosaic Imaging")
        self.start_button.setAutoDefault(False)
        self.start_button.setDefault(False)
        self.start_button.clicked.connect(self.prepare_run)

        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.cancel_mosaic)

        self.af_group = QGroupBox("Autofocus Settings")
        af_layout = QGridLayout(self.af_group)

        self.af_enabled_checkbox = QCheckBox("Enable autofocus")
        self.af_enabled_checkbox.setChecked(True)

        self.af_every_n_label = QLabel("Tiles per autofocus:")
        self.af_every_n_spin = QSpinBox()
        self.af_every_n_spin.setRange(1, 100)
        self.af_every_n_spin.setValue(1)

        self.af_max_steps_label = QLabel("Max steps per autofocus:")
        self.af_max_steps_spin = QSpinBox()
        self.af_max_steps_spin.setRange(1, 100)
        self.af_max_steps_spin.setValue(5)

        self.af_stepsize_label = QLabel("Step Size (μm):")
        self.af_stepsize_spin = QDoubleSpinBox()
        self.af_stepsize_spin.setDecimals(1)
        self.af_stepsize_spin.setRange(0.1, 10.0)
        self.af_stepsize_spin.setValue(0.1)
        self.af_stepsize_spin.setSingleStep(0.1)

        af_layout.addWidget(self.af_enabled_checkbox, 0, 0, 1, 2)
        af_layout.addWidget(self.af_every_n_label, 1, 0); af_layout.addWidget(self.af_every_n_spin, 1, 1)
        af_layout.addWidget(self.af_max_steps_label, 2, 0); af_layout.addWidget(self.af_max_steps_spin, 2, 1)
        af_layout.addWidget(self.af_stepsize_label, 3, 0); af_layout.addWidget(self.af_stepsize_spin, 3, 1)

        self.sidebar_layout.addWidget(self.save_group)
        self.sidebar_layout.addWidget(params_group)
        self.sidebar_layout.addWidget(self.af_group)
        self.sidebar_layout.addWidget(self.start_button)
        self.sidebar_layout.addWidget(self.cancel_button)
        self.sidebar_layout.addStretch()
        main_layout.addWidget(self.sidebar_widget)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.display_label = ZoomableLabel(scroll_area=self.scroll_area)
        self.scroll_area.setWidget(self.display_label)

        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)

        right_layout = QVBoxLayout()
        right_layout.addWidget(self.scroll_area)
        right_layout.addWidget(self.status_label)

        main_layout.addLayout(right_layout, stretch=1)

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Return, Qt.Key_Enter):
            event.ignore()
        else:
            super().keyPressEvent(event)

    def update_status(self, text):
        QTimer.singleShot(0, lambda: self.status_label.setText(text))

    def cancel_mosaic(self):
        self.cancelled = True
        self.update_status("Mosaic acquisition cancelled.")

    def prepare_run(self):
        self.save_dir = self.save_folder_entry.text().strip()
        if not self.save_dir:
            self.update_status("Please select a folder to save results.")
            return
        os.makedirs(self.save_dir, exist_ok=True)
        if self.save_tiles_checkbox.isChecked():
            self.tile_dir = os.path.join(self.save_dir, "tiles")
            os.makedirs(self.tile_dir, exist_ok=True)

        self.cancelled = False
        self._rows = self.rows_spin.value()
        self._cols = self.cols_spin.value()
        self._overlap = self.overlap_spin.value() / 100.0
        self._repetitions = self.repetitions_spin.value()
        self._pattern = self.pattern_combo.currentText()
        self._fov_um = self.fov_um_spin.value()
        self._af_enabled = self.af_enabled_checkbox.isChecked()
        self._af_interval = self.af_every_n_spin.value()
        self._af_max_steps = self.af_max_steps_spin.value()
        self._af_stepsize = int(10 * self.af_stepsize_spin.value())
        self._simulate = self.main_gui.simulation_mode.get()
        self._save_averages = self.save_averages_checkbox.isChecked()
        self._chan = self.main_gui.config["channel_names"][0]

        try:
            port = int(self.main_gui.prior_port_entry.get().strip())
            x0, y0 = get_xy(port)
            self._port, self._x0, self._y0 = port, x0, y0
        except Exception as e:
            self.update_status(f"Stage error: {e}")
            return

        self.tile_w = self.main_gui.config['numsteps_x']
        self.tile_h = self.main_gui.config['numsteps_y']
        self.step_px = int(self.tile_w * (1 - self._overlap))
        self.step_um = int(self._fov_um * (1 - self._overlap))

        self._tile_order = []
        self._tile_colors = {}
        for i in range(self._rows):
            cols = range(self._cols) if (i % 2 == 0 or self._pattern == "Raster") else reversed(range(self._cols))
            for j in cols:
                dx_px = j * self.step_px
                dy_px = i * self.step_px
                dx_um = j * self.step_um
                dy_um = i * self.step_um
                self._tile_order.append((i, j, dx_um, dy_um, dx_px, dy_px))
                self._tile_colors[(i, j)] = QColor(*[random.randint(180,255) for _ in range(3)], 220)

        self.canvas = MosaicCanvas(self.tile_w, self.tile_h, self.step_px, self._rows, self._cols)

        self.worker = MosaicWorker(
            gui=self.main_gui,
            port=self._port,
            x0=self._x0,
            y0=self._y0,
            tile_order=self._tile_order,
            tile_repetitions=self._repetitions,
            af_enabled=self._af_enabled,
            af_interval=self._af_interval,
            af_stepsize=self._af_stepsize,
            af_max_steps=self._af_max_steps,
            simulate=self._simulate,
            chan=self._chan
        )

        self.worker.tile_ready.connect(self.on_tile_ready)
        self.worker.finished.connect(self.on_mosaic_complete)
        self.worker.error.connect(lambda msg: self.update_status(f"Mosaic error: {msg}"))
        self.worker.status_update.connect(self.update_status)

        self.worker.start()
        self.update_status("Starting mosaic...")

    
    @pyqtSlot(int, int, int, int, object)
    def on_tile_ready(self, i, j, dx_px, dy_px, data):
        self.update_status(f"Acquired tile ({i+1},{j+1})...")

        if self.save_tiles_checkbox.isChecked():
            for ch, frame in enumerate(data):
                Image.fromarray(frame).save(os.path.join(self.tile_dir, f"tile_{i}_{j}_ch{ch}.tif"))

        tile_rgb = np.zeros((self.tile_h, self.tile_w, 3), dtype=np.float32)
        visibility = getattr(self.main_gui, 'image_visibility', [True]*len(data))
        colors = getattr(self.main_gui, 'image_colors', [(255,0,0), (0,255,0), (0,0,255)])
        for ch, frame in enumerate(data):
            if ch < len(visibility) and visibility[ch]:
                norm = frame.astype(np.float32)
                for c in range(3):
                    tile_rgb[..., c] += norm * (colors[ch][c] / 255.0)
        tile_rgb = np.clip(tile_rgb, 0, 1)

        self.canvas.blend_tile(i, j, tile_rgb)
        self.update_display()

    @pyqtSlot()
    def on_mosaic_complete(self):
        self.update_status("Mosaic acquisition complete.")
        self.update_display()
        self.save_mosaic()

        stats_df = None
        outpath = os.path.join(self.save_dir, "mosaic_data.csv")

        if self._save_averages:
            stats = self.worker.tile_statistics
            fig, ax = plt.subplots(figsize=(6,4))
            for curve in stats:
                ax.plot(range(1, len(curve)+1), curve, alpha=0.5)

            ax.set_xlabel('Repetition')
            ax.set_ylabel('Average Intensity')
            ax.set_title('Photobleaching Decay Curves')
            fig.tight_layout()

            outpath = os.path.join(self.save_dir, "decay_curves.png")
            fig.savefig(outpath)
            plt.close(fig)

            self.update_status(f"Decay curves saved to:\n{outpath}")

            # stats = self.worker.tile_statistics
            # tile_labels = [f"Tile ({i+1},{j+1})" for i, j, *_ in self.worker.tile_order]
            # stats_df = pd.DataFrame(stats).T  
            # stats_df.columns = tile_labels
            # stats_df.index.name = "Repetition Index"

        # if self.save_focus_metrics_checkbox:
        #     focus_metrics = pd.Series(self.worker.focus_metrics, name="Focus Metric")
        #     if stats_df is not None:
        #         stats_df["Focus Metric"] = focus_metrics.values
        #     else:
        #         stats_df = pd.DataFrame({"Focus Metric": focus_metrics})

        if stats_df is not None:
            stats_df.to_csv(outpath)
            self.update_status(f"Mosaic data saved to:\n{outpath}")

            
        

    def update_display(self):
        image = self.canvas.render_qimage()
        painter = QPainter(image)
        if self.grid_checkbox.isChecked():
            pen = QPen(); pen.setWidth(2); pen.setStyle(Qt.DashLine)
            for (i, j), color in self._tile_colors.items():
                pen.setColor(color)
                painter.setPen(pen)
                painter.drawRect(j * self.step_px, i * self.step_px, self.tile_w, self.tile_h)
        painter.end()
        pix = QPixmap.fromImage(image)
        self.display_label.setPixmap(pix)
        self.display_label.repaint_scaled()


    def browse_save_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Save Folder")
        if folder: self.save_folder_entry.setText(folder)

    def save_mosaic(self):
        if self.save_metadata_checkbox.isChecked():
            md = {
                "rows": self._rows, "cols": self._cols, "overlap": self._overlap,
                "pattern": self._pattern, "fov_um": self._fov_um,
                "initial_position": (self._x0, self._y0),
                "tile_order": [(i, j) for i, j, *_ in self._tile_order]
            }
            with open(os.path.join(self.save_dir, "mosaic_metadata.json"), "w") as f:
                json.dump(md, f, indent=2)

        if self.save_stitched_checkbox.isChecked():
            self.canvas.save_to_file(os.path.join(self.save_dir, "stitched_mosaic.tif"))

        self.update_status("Mosaic saved.")

    def report_memory_estimate(self):
        try:
            tile_w, tile_h = self.main_gui.config['numsteps_x'], self.main_gui.config['numsteps_y']  
            
            rows = self.rows_spin.value()
            cols = self.cols_spin.value()

            mosaic_h = tile_h + (rows - 1) * int(tile_h * (1 - self.overlap_spin.value() / 100.0))
            mosaic_w = tile_w + (cols - 1) * int(tile_w * (1 - self.overlap_spin.value() / 100.0))

            bytes_needed = mosaic_h * mosaic_w * 3 * 4  # float32 RGB
            approx_gb = bytes_needed / (1024**3)

            if approx_gb > 0.2:
                self.update_status(f"WARNING: expected memory usage is {approx_gb:.2f} GB. The program will probably crash.")
            else:
                self.update_status(f"Expected memory usage: {approx_gb:.2f} GB")

        except Exception as e:
            self.update_status(f"Memory estimate failed: {e}")

class MosaicWorker(QThread):
    tile_ready = pyqtSignal(int, int, int, int, object)
    finished = pyqtSignal()
    error = pyqtSignal(str)
    status_update = pyqtSignal(str)

    def __init__(self, gui, port, x0, y0, tile_order, tile_repetitions,
                 af_enabled, af_interval, af_stepsize, af_max_steps,
                 simulate, chan):
        super().__init__()
        self.gui = gui; self.port = port; self.x0 = x0; self.y0 = y0
        self.tile_order = tile_order; self.tile_repetitions = tile_repetitions
        self.af_enabled = af_enabled; self.af_interval = af_interval
        self.af_stepsize = af_stepsize; self.af_max_steps = af_max_steps
        self.simulate = simulate; self.chan = chan
        self.tile_statistics = [[] for _ in range(len(self.tile_order))]
        self.focus_metrics = []

    def run(self):
        try:
            for idx, (i, j, dx_um, dy_um, dx_px, dy_px) in enumerate(self.tile_order):
                move_xy(self.port, self.x0 + dx_um, self.y0 - dy_um)
                if self.af_enabled and idx % self.af_interval == 0 and not self.simulate:
                    self.status_update.emit(f"Autofocusing post-tile ({i+1},{j+1})")
                    z_val, metric = auto_focus(self.gui, self.port, self.chan, step_size=self.af_stepsize, max_steps=self.af_max_steps)
                    self.focus_metrics.append(metric)
                    
                for k in range(self.tile_repetitions):
                    self.status_update.emit(f"Acquiring tile ({i+1},{j+1}), frame {k+1}")
                    acquisition.acquire(self.gui, auxilary=True)

                    data = getattr(self.gui, 'data', []) or []
                    avg = float(np.mean([
                        frame[frame >= 0.25 * frame.max()].mean() if np.any(frame >= 0.25 * frame.max()) else 0
                        for frame in data
                    ]))
                    self.tile_statistics[idx].append(avg)

                data = getattr(self.gui, 'data', []) or []
                self.tile_ready.emit(i, j, dx_px, dy_px, data)
                
            self.finished.emit()

        except Exception as e:
            self.error.emit(str(e))