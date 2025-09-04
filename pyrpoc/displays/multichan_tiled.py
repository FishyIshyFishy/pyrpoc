import numpy as np
import pyqtgraph as pg
from pyqtgraph import ImageView
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PyQt6.QtCore import Qt, pyqtSignal
from .base_display import BaseImageDisplayWidget

class MultichannelImageDisplayWidget(BaseImageDisplayWidget):
    """
    MINIMAL version:
    - Single ImageView
    - Displays the latest frame only
    - Accepts 2D arrays (H, W) or 3D arrays (C, H, W) and shows channel 0
    - Keeps method names/signatures so the rest of the app doesn't break
    """
    display_data_changed = pyqtSignal()

    def __init__(self, app_state, signals):
        super().__init__(app_state, signals)
        self._buffer = None          # optionally store last data for compatibility
        self.current_frame = 0
        self.total_frames = 1
        self.num_channels = 1
        self.channel_names = ['Channel 1']

        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        self.setLayout(layout)

        self.title = QLabel("Display")
        self.title.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        layout.addWidget(self.title)

        self.view = ImageView()
        # keep UI very simple
        if hasattr(self.view.ui, 'roiBtn'):
            self.view.ui.roiBtn.hide()
        if hasattr(self.view.ui, 'menuBtn'):
            self.view.ui.menuBtn.hide()
        if hasattr(self.view.ui, 'histogram'):
            self.view.ui.histogram.hide()
        if hasattr(self.view.ui, 'timeLine'):
            self.view.ui.timeLine.hide()
        if hasattr(self.view.ui, 'roiPlot'):
            self.view.ui.roiPlot.hide()

        self.view.getView().setAspectLocked(True)
        self.view.getView().invertY(True)  # image coords like NumPy
        layout.addWidget(self.view)

    # ---------- Minimal data path ----------

    def prepare_for_acquisition(self, context_or_total_frames):
        # Support either int (number of frames) or a context object (ignored here)
        total_frames = context_or_total_frames if isinstance(context_or_total_frames, int) else 1
        self.total_frames = max(1, int(total_frames))
        self.current_frame = 0
        self._buffer = None
        self.display_data_changed.emit()

    def handle_data_received(self, data):
        """
        Called once per acquired frame. We just display the latest frame.
        Accepts:
          - 2D: (H, W)
          - 3D: (C, H, W) -> uses channel 0
        """
        img = self._to_2d(data)
        if img is None:
            return 0

        # store minimal buffer (last frame only) for compatibility with getters
        self._buffer = img[np.newaxis, np.newaxis, :, :]  # shape (T=1, C=1, H, W)
        self.current_frame = 0
        self.total_frames = 1

        self.view.setImage(img.T, autoLevels=True)
        self.display_data_changed.emit()
        return 1

    def handle_data_updated(self, data: np.ndarray):
        """
        Called when a full dataset is provided (e.g., after load).
        Minimal behavior: show the first frame/channel only.
        Accepts:
          - 3D: (T, H, W)  -> show frame 0
          - 4D: (T, C, H, W)-> show frame 0, channel 0
        """
        if not isinstance(data, np.ndarray):
            return

        if data.ndim == 3:  # (T, H, W)
            T, H, W = data.shape
            self.total_frames = max(1, int(T))
            img = data[0]
        elif data.ndim == 4:  # (T, C, H, W)
            T, C, H, W = data.shape
            self.total_frames = max(1, int(T))
            img = data[0, 0]
        else:
            raise ValueError("Expected 3D (T,H,W) or 4D (T,C,H,W) array")

        # keep a tiny buffer (just the displayed frame) for compatibility
        self._buffer = img[np.newaxis, np.newaxis, :, :]
        self.current_frame = 0

        self.view.setImage(img.T, autoLevels=True)
        self.display_data_changed.emit()

    # ---------- Minimal display helpers ----------

    def display_frame(self, frame_idx):
        """No timeline in minimal version; ignore index and redisplay current frame if available."""
        if self._buffer is not None:
            img = self._buffer[0, 0]
            self.view.setImage(img.T, autoLevels=False)

    def update_display(self):
        """Redraw current image if available."""
        if self._buffer is not None:
            img = self._buffer[0, 0]
            self.view.setImage(img.T, autoLevels=False)

    # ---------- Minimal getters (compatibility) ----------

    def get_current_frame_data(self):
        if self._buffer is None:
            return None
        return self._buffer[0, 0]  # (H, W)

    def get_all_channel_data(self):
        """Return current frame, all channels (we only keep channel 0)."""
        if self._buffer is None:
            return None
        return self._buffer[0]  # shape (C=1, H, W)

    def get_image_data_for_rpoc(self):
        """Same as get_all_channel_data in this minimal build."""
        return self.get_all_channel_data()

    def get_current_frame_index(self):
        return self.current_frame

    def get_display_parameters(self):
        return {
            'num_channels': self.num_channels,
            'channel_names': self.channel_names.copy(),
            'current_frame': self.current_frame
        }

    def set_display_parameters(self, params):
        # No-op in minimal version
        pass

    def refresh_channel_labels(self):
        # No-op in minimal version
        pass

    def _to_2d(self, data):
        """Coerce input to a 2D array to display."""
        if not isinstance(data, np.ndarray):
            return None

        if data.ndim == 2:
            return data
        if data.ndim == 3 and data.shape[0] >= 1:
            return data[0]  # first channel
        # Fallback: try to reshape to square if possible; otherwise reject
        flat = data.reshape(-1)
        size = int(np.sqrt(flat.size))
        if size * size == flat.size:
            return flat.reshape(size, size)
        return None


class MultichannelDisplayParametersWidget(QWidget):
    """
    MINIMAL parameters widget:
    - Just shows a static label so your left panel can mount it.
    - Keeps the same class name/constructor signature.
    """
    def __init__(self, display_widget):
        super().__init__()
        layout = QVBoxLayout(self)
        lbl = QLabel("Display parameters (minimal)")
        layout.addWidget(lbl)
        layout.addStretch()
