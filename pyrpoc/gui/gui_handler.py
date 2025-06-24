from PyQt6.QtCore import QObject, pyqtSignal, QThread
from pyrpoc.imaging.acquisitions import *
import numpy as np

class AppState:
    '''
    APP STATE CLASS
    holds all of the important signaled variables
    '''
    def __init__(self):
        self.modality = 'widefield' 
        self.instruments = { # list of possible instruments, will add as needed
            'galvo': None,
            'daq_inputs': [],
            'tcspc_inputs': []
            } 
        self.acquisition_parameters = {
            'num_frames': 1,
            'x_pixels': 512,
            'y_pixels': 512,
        }
        self.display_parameters = {
            'overlay': True,
        }
        self.rpoc_enabled = False  # Add RPOC enabled state
        self.current_data = None

class StateSignalBus(QObject):
    '''
    SIGNAL COMMUNICATION CLASS
    create all the signals that will be transmitted multi-functionally
    '''
    load_config_btn_clicked = pyqtSignal()
    save_config_btn_clicked = pyqtSignal()

    continuous_btn_clicked = pyqtSignal()
    single_btn_clicked = pyqtSignal()
    stop_btn_clicked = pyqtSignal()

    modality_dropdown_changed = pyqtSignal(str) # emits when the dropdown for the modality is changed
    add_instrument_btn_clicked = pyqtSignal()

    data_updated = pyqtSignal(object) # emits when data acquisition is complete
    console_message = pyqtSignal(str) # emits console status messages

    rpoc_enabled_changed = pyqtSignal(bool) # emits when RPOC enabled checkbox is toggled
    acquisition_parameter_changed = pyqtSignal(str, object) # emits when acquisition parameters change (param_name, new_value)
    
    # Frame-by-frame acquisition signals
    frame_acquired = pyqtSignal(object, int, int) # frame_data, frame_number, total_frames
    acquisition_started = pyqtSignal(int) # total_frames
    acquisition_finished = pyqtSignal()
    
    # Zoom control signals
    # zoom_in = pyqtSignal()
    # zoom_out = pyqtSignal()
    # fit_to_view = pyqtSignal()

    def bind_controllers(self, app_state, main_window):
        self.load_config_btn_clicked.connect(lambda: handle_load_config(app_state))
        self.save_config_btn_clicked.connect(lambda: handle_save_config(app_state))

        self.continuous_btn_clicked.connect(lambda: handle_continuous_acquisition(app_state, self))
        self.single_btn_clicked.connect(lambda: handle_single_acquisition(app_state, self))
        self.stop_btn_clicked.connect(lambda: handle_stop_acquisition(app_state, self))

        self.modality_dropdown_changed.connect(lambda text: handle_modality_changed(text, app_state, main_window))
        self.add_instrument_btn_clicked.connect(lambda: handle_add_instrument(app_state))

        self.data_updated.connect(lambda data: handle_data_updated(data, app_state, main_window))
        self.console_message.connect(lambda message: handle_console_message(message, app_state, main_window))

        self.rpoc_enabled_changed.connect(lambda enabled: handle_rpoc_enabled_changed(enabled, app_state))
        self.acquisition_parameter_changed.connect(lambda param_name, value: handle_acquisition_parameter_changed(param_name, value, app_state))
        
        # Bind frame acquisition signals
        self.frame_acquired.connect(lambda frame_data, frame_num, total_frames: handle_frame_acquired(frame_data, frame_num, total_frames, app_state, main_window))
        self.acquisition_started.connect(lambda total_frames: handle_acquisition_started(total_frames, app_state, main_window))
        self.acquisition_finished.connect(lambda: handle_acquisition_finished(app_state, main_window))
        
        # Bind zoom control signals
        # self.zoom_in.connect(lambda: handle_zoom_in(app_state, main_window))
        # self.zoom_out.connect(lambda: handle_zoom_out(app_state, main_window))
        # self.fit_to_view.connect(lambda: handle_fit_to_view(app_state, main_window))

# Add a worker for threaded acquisition
class AcquisitionWorker(QObject):
    frame_acquired = pyqtSignal(object, int, int)
    acquisition_started = pyqtSignal(int)
    acquisition_finished = pyqtSignal(object)

    def __init__(self, acquisition):
        super().__init__()
        self.acquisition = acquisition
        self._is_running = True

    def run(self):
        self.acquisition.signal_bus = self
        # The acquisition class should handle frame emission and delays
        # If it yields frames, we handle them here; if not, we emit only at the end
        self.acquisition_started.emit(getattr(self.acquisition, 'num_frames', 1))
        result = self.acquisition.perform_acquisition()
        # If perform_acquisition yields frames, collect them
        if hasattr(result, '__iter__') and not isinstance(result, np.ndarray):
            frames = []
            for i, frame in enumerate(result):
                if not self._is_running:
                    break
                frames.append(frame)
                self.frame_acquired.emit(frame, i, len(frames))
            data = np.stack(frames)
        else:
            data = result
        self.acquisition_finished.emit(data)

    def stop(self):
        self._is_running = False

def handle_load_config(app_state):
    return 0

def handle_save_config(app_state):
    return 0

def handle_continuous_acquisition(app_state, signal_bus):
    return 0

def handle_single_acquisition(app_state, signal_bus):
    modality = app_state.modality
    instruments = app_state.instruments
    parameters = app_state.acquisition_parameters
    rpoc_enabled = app_state.rpoc_enabled

    signal_bus.console_message.emit(f"Starting {modality} acquisition...")

    acquisition = None
    match modality:
        case 'simulated':
            try:
                x_pixels = parameters['x_pixels']
                y_pixels = parameters['y_pixels']
                num_frames = parameters['num_frames']
            except Exception as e:
                signal_bus.console_message.emit(f'Error getting parameter values: {e}')
            try:
                acquisition = Simulated(x_pixels, y_pixels, num_frames, signal_bus)
            except Exception as e:
                signal_bus.console_message.emit(f'Error in instantiating Simulated(): {e}')
        case 'widefield':
            try:
                x_pixels = parameters['x_pixels']
                y_pixels = parameters['y_pixels']
                daq_inputs = instruments['daq_inputs']
                tcspc_inputs = instruments['tcspc_inputs']
                signal_bus.console_message.emit(f"Widefield acquisition: {x_pixels}x{y_pixels}")
            except Exception as e:
                signal_bus.console_message.emit(f'Error getting parameter values: {e}')
            try:
                acquisition = Widefield()
            except Exception as e:
                signal_bus.console_message.emit(f'Error in instantiating Simulated(): {e}')
        case 'confocal':
            signal_bus.console_message.emit("Confocal acquisition started")
            acquisition = Confocal()
        case 'mosaic':
            signal_bus.console_message.emit("Mosaic acquisition started")
            acquisition = Mosaic()
        case 'zscan':
            signal_bus.console_message.emit("ZScan acquisition started")
            acquisition = ZScan()
        case 'custom':
            signal_bus.console_message.emit("Custom acquisition started")
            acquisition = Custom()
        case _:
            signal_bus.console_message.emit('Warning: invalid modality, defaulting to simulation')
            acquisition = Simulated()

    if acquisition is not None:
        # Use QThread for all modalities
        worker = AcquisitionWorker(acquisition)
        thread = QThread()
        worker.moveToThread(thread)
        # Connect signals
        worker.frame_acquired.connect(signal_bus.frame_acquired)
        worker.acquisition_started.connect(signal_bus.acquisition_started)
        worker.acquisition_finished.connect(lambda data: _on_acquisition_finished(data, signal_bus))
        thread.started.connect(worker.run)
        thread.start()
        # Store thread/worker to prevent garbage collection
        signal_bus._acq_thread = thread
        signal_bus._acq_worker = worker
    else:
        signal_bus.console_message.emit("Error: Failed to create acquisition object")
    pass

def _on_acquisition_finished(data, signal_bus):
    # Clean up QThread and worker
    if hasattr(signal_bus, '_acq_thread'):
        thread = signal_bus._acq_thread
        worker = signal_bus._acq_worker
        worker.stop()
        thread.quit()
        thread.wait()
        worker.deleteLater()
        thread.deleteLater()
        del signal_bus._acq_thread
        del signal_bus._acq_worker
    signal_bus.acquisition_finished.emit()
    signal_bus.console_message.emit("Acquisition complete!")
    signal_bus.data_updated.emit(data)

def handle_stop_acquisition(app_state):
    # this needs to stop all processes except the main thread i guess
    return 0

def handle_console_message(message, app_state, main_window):
    main_window.top_bar.add_console_message(message)
    return 0

def handle_modality_changed(text, app_state, main_window):
    app_state.modality = text.lower()
    main_window.on_modality_changed(app_state.modality)
    return 0

def handle_data_updated(data, app_state, main_window):
    app_state.current_data = data
    # main_window.mid_layout.update_display(data)  # No longer needed, handled in MiddlePanel
    return 0

def handle_add_instrument(app_state):
    # open the add instrument popout in a side thread and record the necessary params
    return 0

def handle_rpoc_enabled_changed(enabled, app_state):
    app_state.rpoc_enabled = enabled
    return 0

def handle_acquisition_parameter_changed(param_name, value, app_state):
    if param_name in app_state.acquisition_parameters:
        app_state.acquisition_parameters[param_name] = value
    return 0

def handle_frame_acquired(frame_data, frame_num, total_frames, app_state, main_window):
    if not hasattr(app_state, 'frame_data'):
        app_state.frame_data = {}
    app_state.frame_data[frame_num] = frame_data
    
    main_window.mid_layout.update_frame_display(frame_data, frame_num, total_frames)
    return 0

def handle_acquisition_started(total_frames, app_state, main_window):
    app_state.frame_data = {}
    app_state.current_frame = 0

    main_window.mid_layout.setup_frame_controls(total_frames)
    return 0

def handle_acquisition_finished(app_state, main_window):
    """Handle acquisition completion"""
    # Combine all frames into final data
    if hasattr(app_state, 'frame_data') and app_state.frame_data:
        frames = sorted(app_state.frame_data.keys())
        app_state.current_data = np.stack([app_state.frame_data[frame] for frame in frames])
        # main_window.mid_layout.update_display(app_state.current_data)  # Do not overwrite frame-by-frame display
    return 0

def handle_zoom_in(app_state, main_window):
    pass

def handle_zoom_out(app_state, main_window):
    pass

def handle_fit_to_view(app_state, main_window):
    pass