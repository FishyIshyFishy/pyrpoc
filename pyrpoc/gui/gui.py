import numpy as np
from PyQt6.QtWidgets import QApplication, QVBoxLayout, QHBoxLayout, QMainWindow, \
                             QLabel, QWidget, QComboBox, QSplitter, QPushButton, \
                             QPlainTextEdit, QStyle, QGroupBox, QSpinBox, QCheckBox, QLineEdit, QSlider, \
                             QGraphicsView, QGraphicsScene, QGraphicsItem, QGraphicsLineItem, \
                             QFrame, QSizePolicy, QDockWidget
from PyQt6.QtCore import Qt, QPointF, QRectF, QPropertyAnimation, QEasingCurve
from PyQt6.QtGui import QPixmap, QImage, QPen, QBrush, QColor, QPainter, QFont
from pyrpoc.gui.gui_handler import AppState, StateSignalBus
import sys
import pyqtgraph as pg

# ugly red border for outlines
DEV_BORDER_STYLE = """ 
    QWidget {
        border: 2px solid #FF0000;
        border-radius: 4px;
        margin: 2px;
    }
"""
# DEV_BORDER_STYLE = """
#     QWidget {
#         margin: 2px;
#     }
# """

class TopBar(QWidget):
    '''
    horizontal orientation with the important control widgets, most space given to SystemConsole
    AppConfigButtons | AcquisitionControls | SystemStatus
    '''
    def __init__(self, app_state: AppState, signals: StateSignalBus):
        super().__init__()
        self.app_state = app_state
        self.signals = signals
        self.setStyleSheet(DEV_BORDER_STYLE)
        self.setFixedHeight(100) 
        layout = QHBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)

        # two buttons for loading exisitng AppState configs from .json, and saving
        config_widget = QWidget()
        config_layout = QVBoxLayout()

        load_btn = QPushButton('Load Config')
        load_btn.clicked.connect(signals.load_config_btn_clicked.emit)
        config_layout.addWidget(load_btn)

        save_btn = QPushButton('Save Config')
        save_btn.clicked.connect(signals.save_config_btn_clicked.emit)
        config_layout.addWidget(save_btn)

        config_widget.setLayout(config_layout)
        layout.addWidget(config_widget)

        # main acquisition controls: acq continuous, acq single, stop acq
        controls_widget = QWidget()
        controls_layout = QHBoxLayout()
        controls_layout.setContentsMargins(0, 0, 0, 0)

        single_btn = QPushButton()
        single_btn.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))
        single_btn.setToolTip('Start Continuous Acquisition')
        single_btn.clicked.connect(signals.single_btn_clicked.emit)
        controls_layout.addWidget(single_btn)

        continuous_btn = QPushButton()
        continuous_btn.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaSeekForward))
        continuous_btn.setToolTip('Start Single Acquisition')
        continuous_btn.clicked.connect(signals.continuous_btn_clicked.emit)
        controls_layout.addWidget(continuous_btn)

        stop_btn = QPushButton()
        stop_btn.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaStop))
        stop_btn.setToolTip('Stop Acquisition')
        stop_btn.clicked.connect(signals.stop_btn_clicked.emit)
        controls_layout.addWidget(stop_btn)

        controls_widget.setLayout(controls_layout)
        layout.addWidget(controls_widget)

        # system console to display status updates
        console_widget = QWidget()
        console_layout = QVBoxLayout()
        console_layout.setContentsMargins(0, 0, 0, 0)

        self.console = QPlainTextEdit()
        self.console.setReadOnly(True)
        console_layout.addWidget(self.console)

        # tool buttons, eventually i will make these like imageJ icons but for now just text
        self.tool_buttons_widget = QWidget()
        tool_buttons_layout = QHBoxLayout()
        tool_buttons_layout.setContentsMargins(0, 0, 0, 0)
        self.crosshairs_btn = QPushButton('Crosshairs')
        self.crosshairs_btn.setCheckable(True)
        tool_buttons_layout.addWidget(self.crosshairs_btn)
        tool_buttons_layout.addStretch()
        self.tool_buttons_widget.setLayout(tool_buttons_layout)
        console_layout.addWidget(self.tool_buttons_widget)

        console_widget.setLayout(console_layout)
        layout.addWidget(console_widget, stretch=1)

        self.setLayout(layout)

    # called from a signal in gui_handler.py
    def add_console_message(self, message):
        self.console.appendPlainText(message)
        self.console.verticalScrollBar().setValue(self.console.verticalScrollBar().maximum())

class AcquisitionParameters(QWidget):
    def __init__(self, app_state: AppState, signals: StateSignalBus):
        super().__init__()
        self.app_state = app_state
        self.signals = signals
        self.setStyleSheet(DEV_BORDER_STYLE)
        
        main_layout = QVBoxLayout()
        self.group = QGroupBox('Acquisition Parameters')
        self.group.setCheckable(True)
        self.group.setChecked(True)
        
        self.container = QWidget()
        layout = QVBoxLayout()
        
        frames_layout = QHBoxLayout()
        frames_layout.addWidget(QLabel('Number of Frames:'))
        self.frames_spinbox = QSpinBox()
        self.frames_spinbox.setRange(1, 10000)
        self.frames_spinbox.setValue(self.app_state.acquisition_parameters['num_frames'])
        self.frames_spinbox.valueChanged.connect(
            lambda value: self.signals.acquisition_parameter_changed.emit('num_frames', value))
        frames_layout.addWidget(self.frames_spinbox)
        layout.addLayout(frames_layout)
        
        x_pixels_layout = QHBoxLayout()
        x_pixels_layout.addWidget(QLabel('X Pixels:'))
        self.x_pixels_spinbox = QSpinBox()
        self.x_pixels_spinbox.setRange(64, 4096)
        self.x_pixels_spinbox.setValue(self.app_state.acquisition_parameters['x_pixels'])
        self.x_pixels_spinbox.valueChanged.connect(
            lambda value: self.signals.acquisition_parameter_changed.emit('x_pixels', value))
        x_pixels_layout.addWidget(self.x_pixels_spinbox)
        layout.addLayout(x_pixels_layout)
        
        y_pixels_layout = QHBoxLayout()
        y_pixels_layout.addWidget(QLabel('Y Pixels:'))
        self.y_pixels_spinbox = QSpinBox()
        self.y_pixels_spinbox.setRange(64, 4096)
        self.y_pixels_spinbox.setValue(self.app_state.acquisition_parameters['y_pixels'])
        self.y_pixels_spinbox.valueChanged.connect(
            lambda value: self.signals.acquisition_parameter_changed.emit('y_pixels', value))
        y_pixels_layout.addWidget(self.y_pixels_spinbox)
        layout.addLayout(y_pixels_layout)
        
        self.container.setLayout(layout)
        
        group_layout = QVBoxLayout()
        group_layout.addWidget(self.container)
        self.group.setLayout(group_layout)
        
        self.group.toggled.connect(self.container.setVisible)
        
        main_layout.addWidget(self.group)
        self.setLayout(main_layout)


class ModalityControls(QWidget):
    def __init__(self, app_state: AppState, signals: StateSignalBus):
        super().__init__()
        self.app_state = app_state
        self.signals = signals
        self.setStyleSheet(DEV_BORDER_STYLE)
        layout = QHBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)
        ms_label = QLabel('Modality:')
        layout.addWidget(ms_label)
        ms_dropdown = QComboBox()
        ms_dropdown.addItems(['Simulated', 'Widefield', 'Confocal', 'Mosaic', 'ZScan'])
        current_modality = self.app_state.modality.capitalize()
        index = ms_dropdown.findText(current_modality)
        if index >= 0:
            ms_dropdown.setCurrentIndex(index)
        ms_dropdown.currentTextChanged.connect(self.signals.modality_dropdown_changed)
        layout.addWidget(ms_dropdown)
        self.setLayout(layout)


class InstrumentControls(QWidget):
    def __init__(self, app_state: AppState, signals: StateSignalBus):
        super().__init__()
        self.app_state = app_state
        self.signals = signals
        self.setStyleSheet(DEV_BORDER_STYLE)
        main_layout = QVBoxLayout()
        self.group = QGroupBox('Instrument Settings')
        self.group.setCheckable(True)
        self.group.setChecked(True)
        self.container = QWidget()
        layout = QVBoxLayout()

        self.instrument_list = QWidget()
        instrument_list_layout = QVBoxLayout()
        self.instrument_list.setLayout(instrument_list_layout)
        layout.addWidget(self.instrument_list)
        add_btn = QPushButton('Add Instrument')
        add_btn.clicked.connect(self.signals.add_instrument_btn_clicked.emit)
        layout.addWidget(add_btn)
        self.container.setLayout(layout)
        group_layout = QVBoxLayout()
        group_layout.addWidget(self.container)
        self.group.setLayout(group_layout)
        self.group.toggled.connect(self.container.setVisible)
        main_layout.addWidget(self.group)
        self.setLayout(main_layout)


class DisplayControls(QWidget):
    def __init__(self, app_state: AppState, signals: StateSignalBus):
        super().__init__()
        self.app_state = app_state
        self.signals = signals
        self.setStyleSheet(DEV_BORDER_STYLE)
        main_layout = QVBoxLayout()
        self.group = QGroupBox('Display Settings')
        self.group.setCheckable(True)
        self.group.setChecked(True)
        self.container = QWidget()
        layout = QVBoxLayout()
        self.container.setLayout(layout)
        group_layout = QVBoxLayout()
        group_layout.addWidget(self.container)
        self.group.setLayout(group_layout)
        self.group.toggled.connect(self.container.setVisible)
        main_layout.addWidget(self.group)
        self.setLayout(main_layout)


class RightPanel(QWidget):
    def __init__(self, app_state: AppState, signals: StateSignalBus):
        super().__init__()
        self.app_state = app_state
        self.signals = signals
        self.setStyleSheet(DEV_BORDER_STYLE)
        layout = QVBoxLayout()
        
        # RPOC controls
        rpoc_group = QGroupBox('RPOC Controls')
        rpoc_layout = QVBoxLayout()
        rpoc_layout.addWidget(QPushButton('Load Mask'))
        rpoc_layout.addWidget(QPushButton('Save Mask'))
        rpoc_group.setLayout(rpoc_layout)
        layout.addWidget(rpoc_group)
        
        # push it to top for now, this will be easy to work with later ill take the code from the old pyqt gui
        layout.addStretch()
        
        self.setLayout(layout)


class LeftPanel(QWidget):
    def __init__(self, app_state: AppState, signals: StateSignalBus):
        super().__init__()
        self.app_state = app_state
        self.signals = signals
        self.setStyleSheet(DEV_BORDER_STYLE)
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        self.rebuild()

    # necessary for modality changes
    def rebuild(self):
        while self.layout.count():
            child = self.layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        self.modality_controls = ModalityControls(self.app_state, self.signals)
        self.layout.addWidget(self.modality_controls)
        
        self.acquisition_parameters = AcquisitionParameters(self.app_state, self.signals)
        self.layout.addWidget(self.acquisition_parameters)
        
        self.instrument_controls = InstrumentControls(self.app_state, self.signals)
        self.layout.addWidget(self.instrument_controls)
        
        self.display_controls = DisplayControls(self.app_state, self.signals)
        self.layout.addWidget(self.display_controls)
        
        self.layout.addStretch()


########################################################
######### MIDDLE (DISPLAY) PANEL STUFF #################
########################################################

class DockableMiddlePanel(QMainWindow): # needs to be a QMainWindow for the dock widget to work
    def __init__(self, app_state: AppState, signals: StateSignalBus, topbar: TopBar):
        super().__init__()
        self.app_state = app_state
        self.signals = signals
        self.setStyleSheet(DEV_BORDER_STYLE)
       
        central_widget = QWidget()
        layout = QVBoxLayout()

        self.graphics_view = QGraphicsView()
        self.graphics_scene = QGraphicsScene()
        self.graphics_view.setScene(self.graphics_scene)
        self.graphics_view.setMinimumSize(400, 300)
        self.graphics_view.setStyleSheet("""
            QGraphicsView {
                border: 2px dashed #cccccc;
                background-color: #f0f0f0;
            }
        """)
        layout.addWidget(self.graphics_view)

        frame_controls_layout = QHBoxLayout()
        self.frame_label = QLabel('Frame: 0/0')
        frame_controls_layout.addWidget(self.frame_label)
        self.frame_slider = QSlider(Qt.Orientation.Horizontal)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(0)
        self.frame_slider.setValue(0)
        self.frame_slider.setEnabled(False)
        self.frame_slider.valueChanged.connect(self.on_frame_slider_changed)
        frame_controls_layout.addWidget(self.frame_slider)
        layout.addLayout(frame_controls_layout)
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

        self.frame_data = {}
        self.current_frame = 0
        self.total_frames = 0
        self.current_image_item = None
        self.signals.data_updated.connect(self.handle_data_updated)

        self.crosshairs_dock = QDockWidget('Crosshairs', self)
        self.crosshairs_widget = QLabel('Crosshairs controls and display here')
        self.crosshairs_dock.setWidget(self.crosshairs_widget)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.crosshairs_dock)
        self.crosshairs_dock.hide()

        topbar.crosshairs_btn.toggled.connect(self.toggle_crosshairs_dock)

    def toggle_crosshairs_dock(self, checked):
        if checked:
            self.crosshairs_dock.show()
        else:
            self.crosshairs_dock.hide()

    def setup_frame_controls(self, total_frames):
        self.total_frames = total_frames
        self.frame_data = {}
        self.current_frame = 0
        self.frame_slider.setMaximum(max(0, total_frames - 1))
        self.frame_slider.setValue(0)
        self.frame_slider.setEnabled(total_frames > 1)
        self.update_frame_label()
        self.graphics_scene.clear()
        self.current_image_item = None

    def update_frame_label(self):
        self.frame_label.setText(f'Frame: {self.current_frame + 1}/{self.total_frames}')

    def on_frame_slider_changed(self, value):
        self.current_frame = value
        self.update_frame_label()
        if self.current_frame in self.frame_data:
            self.display_frame(self.frame_data[self.current_frame])

    def update_frame_display(self, frame_data, frame_num, total_frames):
        self.frame_data[frame_num] = frame_data
        if frame_num >= self.current_frame:
            self.current_frame = frame_num
            self.frame_slider.setValue(frame_num)
            self.update_frame_label()
            self.display_frame(frame_data)

    def display_frame(self, frame_data):
        if frame_data is None:
            self.graphics_scene.clear()
            self.current_image_item = None
            return
        try:
            if isinstance(frame_data, np.ndarray):
                if frame_data.ndim == 2:
                    height, width = frame_data.shape
                    data_norm = ((frame_data - frame_data.min()) / (frame_data.max() - frame_data.min() + 1e-9) * 255).astype(np.uint8)
                    qimage = QImage(data_norm.data, width, height, width, QImage.Format.Format_Grayscale8)
                else:
                    return
                pixmap = QPixmap.fromImage(qimage)
                self.graphics_scene.clear()
                self.current_image_item = self.graphics_scene.addPixmap(pixmap)
                scene_rect = QRectF(pixmap.rect())
                self.graphics_scene.setSceneRect(scene_rect)
                self.graphics_view.fitInView(self.graphics_scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
            else:
                self.graphics_scene.clear()
                self.current_image_item = None
        except Exception as e:
            print(f"Display error: {e}")
            self.graphics_scene.clear()
            self.current_image_item = None

    def handle_data_updated(self, data):
        if isinstance(data, np.ndarray) and data.ndim == 3:
            last_frame = data.shape[0] - 1
            self.frame_data = {i: data[i] for i in range(data.shape[0])}
            self.total_frames = data.shape[0]
            self.frame_slider.setMaximum(max(0, self.total_frames - 1))
            self.frame_slider.setValue(last_frame)
            self.current_frame = last_frame
            self.update_frame_label()
            self.display_frame(data[last_frame])

        elif isinstance(data, np.ndarray) and data.ndim == 2:
            self.frame_data = {0: data}
            self.total_frames = 1
            self.frame_slider.setMaximum(0)
            self.frame_slider.setValue(0)
            self.current_frame = 0
            self.update_frame_label()
            self.display_frame(data)

        else:
            self.graphics_scene.clear()
            self.current_image_item = None

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.current_image_item:
            self.graphics_view.fitInView(self.graphics_scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)



#########################################
######### MAIN WINDOW #################
#########################################
class MainWindow(QMainWindow):
    def __init__(self, app_state: AppState, signals: StateSignalBus):
        super().__init__()
        self.app_state = app_state
        self.signals = signals
        self.setWindowTitle('pyrpoc - Development Mode')
        self.setGeometry(100, 100, 1400, 900)

        

        central_widget = QWidget()
        central_layout = QVBoxLayout()
        central_layout.setContentsMargins(5, 5, 5, 5)
        central_layout.setSpacing(5)
        
        self.top_bar = TopBar(app_state, signals)
        central_layout.addWidget(self.top_bar, stretch=0)
        
        # Create main splitter for left panel and middle panel
        self.main_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.left_widget = LeftPanel(app_state, signals)
        self.main_splitter.addWidget(self.left_widget)
        
        # Middle panel (dockable)
        self.mid_layout = DockableMiddlePanel(app_state, signals, self.top_bar)
        self.main_splitter.addWidget(self.mid_layout)
        
        # Right panel
        self.right_layout = RightPanel(app_state, signals)
        self.main_splitter.addWidget(self.right_layout)
        
        self.main_splitter.setSizes([200, 800, 200])
        central_layout.addWidget(self.main_splitter, stretch=1)
        
        central_widget.setLayout(central_layout)
        self.setCentralWidget(central_widget)

        self.signals.modality_dropdown_changed.connect(self.on_modality_changed)

    def on_modality_changed(self, new_modality):
        self.left_widget.rebuild()

if __name__ == '__main__':
    app_state = AppState() # can initialize GUI configs with this
    signals = StateSignalBus()

    app = QApplication(sys.argv)
    win = MainWindow(app_state, signals)
    
    signals.bind_controllers(app_state, win)
    win.show()
    sys.exit(app.exec())