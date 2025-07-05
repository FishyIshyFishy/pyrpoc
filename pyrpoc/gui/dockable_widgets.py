from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QSpinBox, QGroupBox, QScrollArea, QSizePolicy
from PyQt6.QtCore import Qt, pyqtSignal, pyqtSlot
import pyqtgraph as pg
import numpy as np

class LinesWidget(QWidget):
    add_mode_requested = pyqtSignal()
    remove_line_requested = pyqtSignal(int)
    line_endpoint_moved = pyqtSignal(int, int, int, int, object)  # line_index, endpoint_idx, x, y, image_data

    def __init__(self, app_state, signals):
        super().__init__()
        self.app_state = app_state
        self.signals = signals
        self.lines = []  # list of line data: [(x1, y1, x2, y2, color), ...]
        self.add_mode = False
        self.color_palette = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8', '#F7DC6F']
        
        layout = QVBoxLayout()
        layout.setContentsMargins(4, 4, 4, 4)  # reduce margins
        layout.setSpacing(2)  # reduce spacing between widgets
        self.setLayout(layout)
        
        # compact top section with button and status in horizontal layout
        top_layout = QHBoxLayout()
        top_layout.setSpacing(4)
        
        # add line button
        self.add_btn = QPushButton("Add Line")
        self.add_btn.setMaximumHeight(24)  # make button more compact
        self.add_btn.clicked.connect(self._emit_add_mode_requested)
        top_layout.addWidget(self.add_btn)
        
        # status display - make it smaller and more compact
        self.status_label = QLabel("Ready")
        self.status_label.setMaximumHeight(20)
        self.status_label.setStyleSheet("QLabel { font-size: 10px; color: #666; }")
        top_layout.addWidget(self.status_label)
        
        layout.addLayout(top_layout)
        
        # single shared plot for all line traces
        self.trace_plot = pg.PlotWidget()
        self.trace_plot.setMinimumHeight(200)
        self.trace_plot.setTitle("Line Traces")
        self.trace_plot.setLabel('left', 'Intensity')
        self.trace_plot.setLabel('bottom', 'Position along line')
        layout.addWidget(self.trace_plot)
        
        # scrollable area for line controls
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setSizePolicy(scroll_area.sizePolicy().horizontalPolicy(), QSizePolicy.Policy.Expanding)
        scroll_area.setFrameStyle(0)  # remove frame to save space
        
        self.lines_container = QWidget()
        self.lines_layout = QVBoxLayout()
        self.lines_layout.setSpacing(2)  # reduce spacing between lines
        self.lines_container.setLayout(self.lines_layout)
        
        scroll_area.setWidget(self.lines_container)
        layout.addWidget(scroll_area, stretch=1)

    def _emit_add_mode_requested(self):
        self.add_mode = True
        self.status_label.setText("Click to place line endpoints")
        self.add_btn.setText("Cancel Add")
        self.add_btn.clicked.disconnect()
        self.add_btn.clicked.connect(self.cancel_add_mode)
        self.add_mode_requested.emit()

    def cancel_add_mode(self):
        self.add_mode = False
        self.status_label.setText("Ready")
        self.add_btn.setText("Add Line")
        self.add_btn.clicked.disconnect()
        self.add_btn.clicked.connect(self._emit_add_mode_requested)

    @pyqtSlot(int, int, int, int, object, list)
    def add_line(self, x1, y1, x2, y2, image_data=None, channel_names=None):
        if not self.add_mode:
            return
        
        # assign unique color to each line by cycling through palette
        color = self.color_palette[len(self.lines) % len(self.color_palette)]
        line_data = {
            'x1': x1,
            'y1': y1,
            'x2': x2,
            'y2': y2,
            'color': color,
            'index': len(self.lines),
            'channel_names': channel_names or ['Channel 1']
        }
        self.lines.append(line_data)
        self.add_line_ui(line_data)
        self.cancel_add_mode()
        if image_data is not None:
            self.update_all_traces(image_data)

    @pyqtSlot(int, int, int, int, object)
    def update_line_endpoint(self, line_index, endpoint_idx, x, y, image_data=None):
        if 0 <= line_index < len(self.lines):
            line = self.lines[line_index]
            if endpoint_idx == 0:
                line['x1'] = x
                line['y1'] = y
            else:
                line['x2'] = x
                line['y2'] = y
            
            # update ui
            group = self.lines_layout.itemAt(line_index).widget()
            if group:
                pos_label = group.layout().itemAt(0).layout().itemAt(0).widget()
                pos_label.setText(f"Line {line_index + 1}: ({line['x1']:.1f},{line['y1']:.1f}) to ({line['x2']:.1f},{line['y2']:.1f})")
            
            if image_data is not None:
                self.update_all_traces(image_data)

    @pyqtSlot(int)
    def remove_line(self, index):
        if 0 <= index < len(self.lines):
            widget = self.lines_layout.itemAt(index).widget()
            if widget:
                widget.deleteLater()
                self.lines_layout.removeItem(self.lines_layout.itemAt(index))
            self.lines.pop(index)
            
            for i, line in enumerate(self.lines):
                line['index'] = i
                line['color'] = self.color_palette[i % len(self.color_palette)]
            
            self.remove_line_requested.emit(index)

            # update remaining line uis
            for i, line in enumerate(self.lines):
                group = self.lines_layout.itemAt(i).widget()
                if group:
                    pos_label = group.layout().itemAt(0).layout().itemAt(0).widget()
                    pos_label.setText(f"Line {i + 1}: ({line['x1']:.1f},{line['y1']:.1f}) to ({line['x2']:.1f},{line['y2']:.1f})")

                    group.setStyleSheet(f"QGroupBox {{ border: 2px solid {line['color']}; }}")
            
            # update the plot to remove the trace
            self.update_all_traces(self.app_state.current_data)

    @pyqtSlot(object)
    def update_all_traces(self, image_data):
        if image_data is None:
            return
        
        self.trace_plot.clear()
        
        if isinstance(image_data, np.ndarray) and image_data.ndim == 3 and image_data.shape[0] > 1:
            num_channels = image_data.shape[0]
            
            for line_data in self.lines:
                channel_names = line_data.get('channel_names', [f'Ch{i+1}' for i in range(num_channels)])
                for ch_idx in range(num_channels):
                    channel_data = image_data[ch_idx]
                    trace = self.get_line_trace(line_data, channel_data)
                    if trace is not None:
                        positions = np.arange(len(trace))
                        
                        # Use different line styles for different lines and channels
                        # Cycle through styles: Solid, Dash, Dot, DashDot
                        style_cycle = [
                            pg.QtCore.Qt.PenStyle.SolidLine,
                            pg.QtCore.Qt.PenStyle.DashLine,
                            pg.QtCore.Qt.PenStyle.DotLine,
                            pg.QtCore.Qt.PenStyle.DashDotLine
                        ]
                        line_style = style_cycle[line_data['index'] % len(style_cycle)]
                        
                        # Use different line styles for different channels within the same line
                        if ch_idx > 0:  # Additional channels get different style
                            # Shift the style for additional channels
                            style_index = (line_data['index'] + ch_idx) % len(style_cycle)
                            line_style = style_cycle[style_index]
                        
                        pen = pg.mkPen(line_data['color'], width=2, style=line_style)
                        
                        # Use actual channel name if available, otherwise fallback
                        channel_name = channel_names[ch_idx] if ch_idx < len(channel_names) else f'Ch{ch_idx+1}'
                        self.trace_plot.plot(positions, trace, 
                                           pen=pen,
                                           name=f"Line {line_data['index'] + 1} {channel_name}")
        else:
            for line_data in self.lines:
                trace = self.get_line_trace(line_data, image_data)
                if trace is not None:
                    positions = np.arange(len(trace))
                    channel_names = line_data.get('channel_names', ['Channel 1'])
                    channel_name = channel_names[0] if channel_names else 'Channel 1'
                    
                    # Use different line styles for different lines
                    # Cycle through styles: Solid, Dash, Dot, DashDot
                    style_cycle = [
                        pg.QtCore.Qt.PenStyle.SolidLine,
                        pg.QtCore.Qt.PenStyle.DashLine,
                        pg.QtCore.Qt.PenStyle.DotLine,
                        pg.QtCore.Qt.PenStyle.DashDotLine
                    ]
                    line_style = style_cycle[line_data['index'] % len(style_cycle)]
                    
                    self.trace_plot.plot(positions, trace, 
                                       pen=pg.mkPen(line_data['color'], width=2, style=line_style),
                                       name=f"Line {line_data['index'] + 1} {channel_name}")
        
        # Show legend
        self.trace_plot.addLegend()
        
        # Configure legend for better visibility
        if hasattr(self.trace_plot, 'legend'):
            self.trace_plot.legend.setBrush(pg.mkBrush('w'))
            self.trace_plot.legend.setPen(pg.mkPen('k'))

    def get_line_trace(self, line_data, image_data):
        """extract intensity values along a line using bresenham's algorithm"""
        if image_data is None:
            return None
            
        if isinstance(image_data, np.ndarray):
            if image_data.ndim == 3:
                current_frame = 0  # should be passed in for multi-frame
                if 0 <= current_frame < image_data.shape[0]:
                    frame_data = image_data[current_frame]
                else:
                    return None
            else:
                frame_data = image_data
        else:
            return None
            
        height, width = frame_data.shape
        x1, y1 = int(line_data['x1']), int(line_data['y1'])
        x2, y2 = int(line_data['x2']), int(line_data['y2'])
        
        # clamp coordinates to image bounds
        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(0, min(x2, width - 1))
        y2 = max(0, min(y2, height - 1))
        
        # bresenham's line algorithm
        points = []
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1
        err = dx - dy
        
        x, y = x1, y1
        while True:
            if 0 <= x < width and 0 <= y < height:
                points.append(frame_data[y, x])
            if x == x2 and y == y2:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy
        
        return np.array(points) if points else None

    def add_line_ui(self, line_data):
        group = QGroupBox(f"Line {line_data['index'] + 1}")
        group.setStyleSheet(f"QGroupBox {{ border: 2px solid {line_data['color']}; }}")
        layout = QVBoxLayout()
        group.setLayout(layout)
        
        pos_layout = QHBoxLayout()
        pos_label = QLabel(f"Line {line_data['index'] + 1}: ({line_data['x1']:.1f},{line_data['y1']:.1f}) to ({line_data['x2']:.1f},{line_data['y2']:.1f})")
        pos_layout.addWidget(pos_label)
        
        remove_btn = QPushButton("Remove")
        remove_btn.clicked.connect(lambda: self.remove_line(line_data['index']))
        pos_layout.addWidget(remove_btn)
        
        layout.addLayout(pos_layout)
        self.lines_layout.addWidget(group)

    def get_lines(self):
        return [(line['x1'], line['y1'], line['x2'], line['y2'], line['color']) for line in self.lines]

    def update_status(self, enabled):
        if enabled:
            self.status_label.setText("Ready")
            self.setEnabled(True)
        else:
            self.status_label.setText("Lines: Disabled")
            self.setEnabled(False)
            self.cancel_add_mode() 