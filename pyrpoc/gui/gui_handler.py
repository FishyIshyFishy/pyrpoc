from PyQt6.QtCore import QObject, pyqtSignal, QThread
from PyQt6.QtWidgets import QFileDialog, QMessageBox
from pyrpoc.imaging.acquisitions import *
from pyrpoc.imaging.instruments import create_instrument, get_instruments_by_type
import numpy as np
import json
import os
from pathlib import Path

class AppState:
    '''
    APP STATE CLASS
    holds all of the important signaled variables
    '''
    def __init__(self):
        self.modality = 'simulated' 
        self.instruments = {}  # instrument_id -> instrument object
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
        
        # RPOC masks storage
        self.rpoc_masks = {}

    def get_instruments_by_type(self, instrument_type):
        """Get all instruments of a specific type"""
        return get_instruments_by_type(self.instruments, instrument_type)
    
    def serialize_instruments(self):
        """Convert instruments to serializable format for saving"""
        serialized = {}
        for instrument_id, instrument in self.instruments.items():
            serialized[instrument_id] = {
                'name': instrument.name,
                'instrument_type': instrument.instrument_type,
                'parameters': instrument.parameters.copy(),
                'connected': instrument.connected
            }
        return serialized
    
    def deserialize_instruments(self, serialized_instruments):
        """Recreate instruments from serialized format"""
        self.instruments.clear()
        for instrument_id, data in serialized_instruments.items():
            try:
                # Ensure name is a string, not a dict
                instrument_name = data.get('name', 'Unknown Instrument')
                if isinstance(instrument_name, dict):
                    # If name is somehow a dict, try to extract a string value
                    instrument_name = str(instrument_name)
                
                # Ensure parameters is a dict
                parameters = data.get('parameters', {})
                if not isinstance(parameters, dict):
                    parameters = {}
                
                instrument = create_instrument(
                    data['instrument_type'], 
                    instrument_name, 
                    parameters
                )
                instrument.connected = data.get('connected', False)
                self.instruments[int(instrument_id)] = instrument
            except Exception as e:
                print(f"Failed to recreate instrument {data.get('name', 'Unknown')}: {e}")

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
    add_modality_instrument = pyqtSignal(str) # emits when adding a modality-specific instrument (instrument_type)
    instrument_removed = pyqtSignal(int) # emits when an instrument is removed (instrument_id)

    data_updated = pyqtSignal(object) # emits when data acquisition is complete
    console_message = pyqtSignal(str) # emits console status messages

    rpoc_enabled_changed = pyqtSignal(bool) # emits when RPOC enabled checkbox is toggled
    acquisition_parameter_changed = pyqtSignal(str, object) # emits when acquisition parameters change (param_name, new_value)
    
    # UI state signals
    ui_state_changed = pyqtSignal(str, object) # emits when UI state changes (param_name, new_value)
    lines_toggled = pyqtSignal(bool) # emits when lines are toggled
    
    # acquisition signals (agnostic)
    frame_acquired = pyqtSignal(object, int, int) # data_unit, index, total

    # RPOC mask creation
    mask_created = pyqtSignal(object) # emits when a mask is created
    rpoc_channel_removed = pyqtSignal(int) # emits when an RPOC channel is removed (channel_id)

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
        
        self.load_config_btn_clicked.connect(lambda: handle_load_config_with_ui_update(app_state, main_window))
        self.save_config_btn_clicked.connect(lambda: handle_save_config(app_state))

        self.continuous_btn_clicked.connect(lambda: handle_continuous_acquisition(app_state, self))
        self.single_btn_clicked.connect(lambda: handle_single_acquisition(app_state, self))
        self.stop_btn_clicked.connect(lambda: handle_stop_acquisition(app_state, self))

        self.modality_dropdown_changed.connect(lambda text: handle_modality_changed(text, app_state, main_window))
        self.add_instrument_btn_clicked.connect(lambda: handle_add_instrument(app_state, main_window))
        self.add_modality_instrument.connect(lambda instrument_type: handle_add_modality_instrument(instrument_type, app_state, main_window))
        self.instrument_removed.connect(lambda instrument_id: handle_instrument_removed(instrument_id, app_state))

        self.data_updated.connect(lambda data: handle_data_updated(data, app_state, main_window))
        self.console_message.connect(lambda message: handle_console_message(message, app_state, main_window))

        self.rpoc_enabled_changed.connect(lambda enabled: handle_rpoc_enabled_changed(enabled, app_state))
        self.acquisition_parameter_changed.connect(lambda param_name, value: handle_acquisition_parameter_changed(param_name, value, app_state))
        
        self.ui_state_changed.connect(lambda param_name, value: handle_ui_state_changed(param_name, value, app_state))
        self.lines_toggled.connect(lambda enabled: handle_lines_toggled(enabled, app_state))
        
        self.frame_acquired.connect(lambda data_unit, idx, total: handle_frame_acquired(data_unit, idx, total, app_state, main_window))

        self.mask_created.connect(lambda mask: handle_mask_created(mask, app_state, main_window, self))
        self.rpoc_channel_removed.connect(lambda channel_id: handle_rpoc_channel_removed(channel_id, app_state, self))

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
    """Load configuration from JSON file"""
    try:
        file_path, _ = QFileDialog.getOpenFileName(
            None, 'Load Configuration', '', 
            'JSON files (*.json);;All files (*)'
        )
        if not file_path:
            return 0
            
        with open(file_path, 'r') as f:
            config_data = json.load(f)
        
        # Load basic parameters
        if 'modality' in config_data:
            app_state.modality = config_data['modality']
        if 'acquisition_parameters' in config_data:
            app_state.acquisition_parameters.update(config_data['acquisition_parameters'])
        if 'display_parameters' in config_data:
            app_state.display_parameters.update(config_data['display_parameters'])
        if 'rpoc_enabled' in config_data:
            app_state.rpoc_enabled = config_data['rpoc_enabled']
        if 'ui_state' in config_data:
            app_state.ui_state.update(config_data['ui_state'])
        
        # Load instruments
        if 'instruments' in config_data:
            app_state.deserialize_instruments(config_data['instruments'])
        
        # Load RPOC masks (if they exist)
        if 'rpoc_masks' in config_data:
            app_state.rpoc_masks = config_data['rpoc_masks']
        
        print(f"Configuration loaded from {file_path}")
        return 1
        
    except Exception as e:
        QMessageBox.critical(None, "Load Error", f"Failed to load configuration: {e}")
        return 0

def handle_load_config_with_ui_update(app_state, main_window):
    """Load configuration and update UI accordingly"""
    result = handle_load_config(app_state)
    if result == 1:
        # Update UI to reflect loaded configuration
        main_window.rebuild_gui()
        return 1
    return 0

def handle_save_config(app_state):
    """Save configuration to JSON file"""
    try:
        file_path, _ = QFileDialog.getSaveFileName(
            None, 'Save Configuration', 'config.json', 
            'JSON files (*.json);;All files (*)'
        )
        if not file_path:
            return 0
        
        # Prepare configuration data
        config_data = {
            'modality': app_state.modality,
            'acquisition_parameters': app_state.acquisition_parameters.copy(),
            'display_parameters': app_state.display_parameters.copy(),
            'rpoc_enabled': app_state.rpoc_enabled,
            'ui_state': app_state.ui_state.copy(),
            'instruments': app_state.serialize_instruments(),
            'rpoc_masks': app_state.rpoc_masks.copy()
        }
        
        # Save to file
        with open(file_path, 'w') as f:
            json.dump(config_data, f, indent=2)
        
        print(f"Configuration saved to {file_path}")
        return 1
        
    except Exception as e:
        QMessageBox.critical(None, "Save Error", f"Failed to save configuration: {e}")
        return 0

def handle_continuous_acquisition(app_state, signal_bus):
    return 0

def handle_single_acquisition(app_state, signal_bus):
    """Start single acquisition with proper parameter validation and instrument handling"""
    modality = app_state.modality
    parameters = app_state.acquisition_parameters.copy()  # Take snapshot of parameters
    rpoc_enabled = app_state.rpoc_enabled

    signal_bus.console_message.emit(f"Starting {modality} acquisition...")

    # Validate parameters before starting acquisition
    try:
        validate_acquisition_parameters(parameters, modality)
    except ValueError as e:
        signal_bus.console_message.emit(f"Parameter validation error: {e}")
        return

    # Get instruments organized by type for acquisition
    instruments_by_type = {}
    for instrument_type in ['Galvo', 'Data Input', 'Delay Stage', 'Zaber Stage']:
        instruments_by_type[instrument_type] = app_state.get_instruments_by_type(instrument_type)

    acquisition = None
    try:
        match modality:
            case 'simulated':
                x_pixels = parameters['x_pixels']
                y_pixels = parameters['y_pixels']
                num_frames = parameters['num_frames']
                acquisition = Simulated(x_pixels, y_pixels, num_frames, signal_bus)
                
            case 'widefield':
                x_pixels = parameters['x_pixels']
                y_pixels = parameters['y_pixels']
                # Get instruments by type instead of by ID
                data_inputs = instruments_by_type.get('Data Input', [])
                signal_bus.console_message.emit(f"Widefield acquisition: {x_pixels}x{y_pixels}")
                acquisition = Widefield(data_inputs=data_inputs)
                
            case 'confocal':
                signal_bus.console_message.emit("Confocal acquisition started")
                # Pass organized instruments to confocal acquisition
                galvos = instruments_by_type.get('Galvo', [])
                data_inputs = instruments_by_type.get('Data Input', [])
                acquisition = Confocal(galvos=galvos, data_inputs=data_inputs)
                
            case 'mosaic':
                signal_bus.console_message.emit("Mosaic acquisition started")
                # Pass stage instruments to mosaic acquisition
                stages = instruments_by_type.get('Delay Stage', []) + instruments_by_type.get('Zaber Stage', [])
                acquisition = Mosaic(stages=stages)
                
            case 'zscan':
                signal_bus.console_message.emit("ZScan acquisition started")
                stages = instruments_by_type.get('Delay Stage', []) + instruments_by_type.get('Zaber Stage', [])
                acquisition = ZScan(stages=stages)
                
            case 'custom':
                signal_bus.console_message.emit("Custom acquisition started")
                acquisition = Custom()
                
            case _:
                signal_bus.console_message.emit('Warning: invalid modality, defaulting to simulation')
                acquisition = Simulated()

    except Exception as e:
        signal_bus.console_message.emit(f'Error creating acquisition object: {e}')
        return

    if acquisition is not None:
        # Note: Acquisition objects receive parameters through their constructors
        # and instruments are passed as needed. The clean parameter snapshot
        # has already been validated and passed to the acquisition constructor.
        
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

def validate_acquisition_parameters(parameters, modality):
    """Validate acquisition parameters before starting acquisition"""
    required_params = {
        'simulated': ['x_pixels', 'y_pixels', 'num_frames'],
        'widefield': ['x_pixels', 'y_pixels'],
        'confocal': ['x_pixels', 'y_pixels'],
        'mosaic': ['x_pixels', 'y_pixels'],
        'zscan': ['x_pixels', 'y_pixels'],
        'custom': ['x_pixels', 'y_pixels']
    }
    
    if modality not in required_params:
        raise ValueError(f"Unknown modality: {modality}")
    
    missing_params = []
    for param in required_params[modality]:
        if param not in parameters:
            missing_params.append(param)
    
    if missing_params:
        raise ValueError(f"Missing required parameters for {modality}: {missing_params}")
    
    # Validate parameter values
    if 'x_pixels' in parameters and (parameters['x_pixels'] <= 0 or parameters['x_pixels'] > 10000):
        raise ValueError("x_pixels must be between 1 and 10000")
    
    if 'y_pixels' in parameters and (parameters['y_pixels'] <= 0 or parameters['y_pixels'] > 10000):
        raise ValueError("y_pixels must be between 1 and 10000")
    
    if 'num_frames' in parameters and (parameters['num_frames'] <= 0 or parameters['num_frames'] > 10000):
        raise ValueError("num_frames must be between 1 and 10000")

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

def handle_add_instrument(app_state, main_window):
    # Show dialog to select instrument type
    from PyQt6.QtWidgets import QDialog, QVBoxLayout, QComboBox, QPushButton, QLabel
    from pyrpoc.imaging.instruments import InstrumentDialog, create_instrument
    
    dialog = QDialog(main_window)
    dialog.setWindowTitle("Add Instrument")
    dialog.setModal(True)
    
    layout = QVBoxLayout()
    
    layout.addWidget(QLabel("Select instrument type:"))
    
    combo = QComboBox()
    combo.addItems(['Delay Stage', 'Zaber Stage'])
    layout.addWidget(combo)
    
    button_layout = QHBoxLayout()
    ok_btn = QPushButton("OK")
    cancel_btn = QPushButton("Cancel")
    
    ok_btn.clicked.connect(dialog.accept)
    cancel_btn.clicked.connect(dialog.reject)
    
    button_layout.addWidget(ok_btn)
    button_layout.addWidget(cancel_btn)
    layout.addLayout(button_layout)
    
    dialog.setLayout(layout)
    
    if dialog.exec() == QDialog.DialogCode.Accepted:
        instrument_type = combo.currentText()
        handle_add_modality_instrument(instrument_type, app_state, main_window)
    
    return 0

def handle_add_modality_instrument(instrument_type, app_state, main_window):
    from pyrpoc.imaging.instruments import InstrumentDialog, create_instrument
    
    # Show configuration dialog
    dialog = InstrumentDialog(instrument_type, main_window)
    if dialog.exec() == QDialog.DialogCode.Accepted:
        parameters = dialog.get_parameters()
        
        # Create instrument
        if parameters is not None:  # Check if validation passed
            instrument_name = parameters.get('name', f'{instrument_type}')
            instrument = create_instrument(instrument_type, instrument_name, parameters)
            
            # Ensure instrument name is properly set
            if hasattr(instrument, 'name') and instrument.name:
                display_name = instrument.name
            else:
                display_name = instrument_name
        else:
            main_window.signals.console_message.emit(f"Failed to create {instrument_type} - invalid parameters")
            return 0
        
        # Initialize connection
        if instrument.initialize():
            # Add to app_state
            if not hasattr(app_state, 'instruments'):
                app_state.instruments = {}
            
            # Generate unique ID
            instrument_id = 1
            while instrument_id in app_state.instruments:
                instrument_id += 1
            
            app_state.instruments[instrument_id] = instrument
            
            # Add to GUI
            if hasattr(main_window, 'left_widget') and hasattr(main_window.left_widget, 'instrument_controls'):
                main_window.left_widget.instrument_controls.add_instrument(instrument_id, instrument)
                # Update modality buttons to hide the one we just added
                main_window.left_widget.instrument_controls.update_modality_buttons()
            
            main_window.signals.console_message.emit(f"Added {display_name} successfully")
        else:
            main_window.signals.console_message.emit(f"Failed to connect to {display_name}")
    
    return 0

def handle_instrument_removed(instrument_id, app_state):
    if hasattr(app_state, 'instruments') and instrument_id in app_state.instruments:
        del app_state.instruments[instrument_id]
        # Update modality buttons to show the button for the removed instrument type
        # This will be handled by the GUI when it rebuilds
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

def handle_mask_created(mask, app_state, main_window, signal_bus):
    # the mask is now handled by the individual channel widgets
    # this function is kept for backward compatibility but may not be used
    signal_bus.console_message.emit("RPOC mask created.")
    return 0

def handle_rpoc_channel_removed(channel_id, app_state, signal_bus):
    # remove the mask from app_state if it exists
    if hasattr(app_state, 'rpoc_masks') and channel_id in app_state.rpoc_masks:
        del app_state.rpoc_masks[channel_id]
    signal_bus.console_message.emit(f"RPOC channel {channel_id} removed.")
    return 0