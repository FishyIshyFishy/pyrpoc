from PyQt6.QtCore import QObject, pyqtSignal, QThread
from pyrpoc.imaging.acquisitions import *
import numpy as np

class AppState:
    '''
    APP STATE CLASS
    holds all of the important signaled variables
    '''
    def __init__(self):
        self.modality = 'simulated' 
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
            # overlay parameter removed - not implemented yet
        }
        self.rpoc_enabled = False  # Add RPOC enabled state
        self.current_data = None
        
        # UI state parameters
        self.ui_state = {
            'acquisition_parameters_visible': True,
            'instrument_controls_visible': True,
            'display_controls_visible': True,
            'lines_enabled': False,
            'main_splitter_sizes': [200, 800, 200],  # left, middle, right panel sizes
        }

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
    
    # UI state signals
    ui_state_changed = pyqtSignal(str, object) # emits when UI state changes (param_name, new_value)
    lines_toggled = pyqtSignal(bool) # emits when lines are toggled
    
    # acquisition signals (agnostic)
    frame_acquired = pyqtSignal(object, int, int) # data_unit, index, total

    def __init__(self):
        super().__init__()
        self._connected = False

    def disconnect_all(self):
        if self._connected:
            self.disconnect()
            self._connected = False

    def bind_controllers(self, app_state, main_window):
        # recalls when gui is rebuilt on modality changes
        # need to disconnect the old connections
        self.disconnect_all()
        
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
        
        self.ui_state_changed.connect(lambda param_name, value: handle_ui_state_changed(param_name, value, app_state))
        self.lines_toggled.connect(lambda enabled: handle_lines_toggled(enabled, app_state))
        
        self.frame_acquired.connect(lambda data_unit, idx, total: handle_frame_acquired(data_unit, idx, total, app_state, main_window))

        # Lines <-> Image Display signal wiring
        image_display = main_window.mid_layout.image_display_widget
        lines = main_window.mid_layout.lines_widget

        # Use the base class method to handle all signal connections
        image_display.connect_lines_widget(lines)
        
        self._connected = True

class AcquisitionWorker(QObject):
    frame_acquired = pyqtSignal(object, int, int)
    acquisition_finished = pyqtSignal(object)

    def __init__(self, acquisition):
        super().__init__()
        self.acquisition = acquisition
        self._is_running = True

    def run(self):
        # Pass the stop flag to the acquisition
        self.acquisition.set_stop_flag(lambda: not self._is_running)
        
        # The acquisition should emit frame_acquired as it acquires data
        result = self.acquisition.perform_acquisition()
        if self._is_running:  # Only emit if not stopped
            self.acquisition_finished.emit(result)

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
        worker = AcquisitionWorker(acquisition)
        thread = QThread()
        worker.moveToThread(thread)

        worker.frame_acquired.connect(signal_bus.frame_acquired)
        worker.acquisition_finished.connect(lambda data: handle_acquisition_thread_finished(data, signal_bus, thread, worker))
        thread.started.connect(worker.run)
        thread.start()

        # garbage collection
        signal_bus._acq_thread = thread
        signal_bus._acq_worker = worker
    else:
        signal_bus.console_message.emit("Error: Failed to create acquisition object")
    pass

def handle_acquisition_thread_finished(data, signal_bus, thread, worker):
    # Clean up thread and worker after acquisition is finished
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
    signal_bus.console_message.emit("Acquisition complete!")
    signal_bus.data_updated.emit(data)

def handle_stop_acquisition(app_state, signal_bus):
    """Stop any running acquisition threads"""
    if hasattr(signal_bus, '_acq_thread') and hasattr(signal_bus, '_acq_worker'):
        thread = signal_bus._acq_thread
        worker = signal_bus._acq_worker
        
        # Stop the worker
        worker.stop()
        
        # Quit and wait for thread to finish
        thread.quit()
        if thread.wait(5000):  # Wait up to 5 seconds
            thread.deleteLater()
            worker.deleteLater()
        else:
            # Force termination if thread doesn't quit gracefully
            thread.terminate()
            thread.wait()
            thread.deleteLater()
            worker.deleteLater()
        
        # Clean up references
        del signal_bus._acq_thread
        del signal_bus._acq_worker
        
        signal_bus.console_message.emit("Acquisition stopped")
    return 0

def cleanup_acquisition_threads(signal_bus):
    """Clean up any running acquisition threads"""
    if hasattr(signal_bus, '_acq_thread') and hasattr(signal_bus, '_acq_worker'):
        thread = signal_bus._acq_thread
        worker = signal_bus._acq_worker
        
        # Stop the worker
        worker.stop()
        
        # Quit and wait for thread to finish
        thread.quit()
        if thread.wait(1000):  # Wait up to 1 second
            thread.deleteLater()
            worker.deleteLater()
        else:
            # Force termination if thread doesn't quit gracefully
            thread.terminate()
            thread.wait()
            thread.deleteLater()
            worker.deleteLater()
        
        # Clean up references
        del signal_bus._acq_thread
        del signal_bus._acq_worker

def handle_console_message(message, app_state, main_window):
    main_window.top_bar.add_console_message(message)
    return 0

def handle_modality_changed(text, app_state, main_window):
    app_state.modality = text.lower()
    
    # Clean up any running acquisition threads before rebuilding
    cleanup_acquisition_threads(main_window.signals)
    
    # Clear current data to prevent stale data from showing
    app_state.current_data = None
    
    # Rebuild the entire GUI
    main_window.rebuild_gui()
    
    # Reconnect all signals after rebuild
    main_window.signals.bind_controllers(app_state, main_window)
    
    return 0

def handle_data_updated(data, app_state, main_window):
    # TODO: fix the signaling to have the general display be compatible with all modalities
    # as of right now its only for multiframe 2D data
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

def handle_ui_state_changed(param_name, value, app_state):
    if param_name in app_state.ui_state:
        app_state.ui_state[param_name] = value
    return 0

def handle_lines_toggled(enabled, app_state):
    app_state.ui_state['lines_enabled'] = enabled
    return 0

def handle_frame_acquired(data_unit, idx, total, app_state, main_window):
    # Just route the data to the widget, don't interpret it
    if hasattr(main_window, 'mid_layout') and hasattr(main_window.mid_layout, 'image_display_widget'):
        widget = main_window.mid_layout.image_display_widget
        if hasattr(widget, 'handle_frame_acquired'):
            widget.handle_frame_acquired(data_unit, idx, total)
    return 0

def handle_zoom_in(app_state, main_window):
    pass

def handle_zoom_out(app_state, main_window):
    pass

def handle_fit_to_view(app_state, main_window):
    pass