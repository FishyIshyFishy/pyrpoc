import json
from PyQt6.QtWidgets import QFileDialog, QMessageBox
from pyrpoc.instruments.instrument_manager import create_instrument, get_instruments_by_type, show_add_instrument_dialog, show_configure_instrument_dialog
from pyrpoc.acquisitions import *
from PyQt6.QtCore import QObject, pyqtSignal, QThread
import numpy as np
import os
from pathlib import Path

class AppState:
    '''
    holds all of the important signaled variables as well as things for loading/saving configs 
    TODO: reclaim appstate after closure
    '''
    def __init__(self):
        self.modality = 'simulated' 
        self.instruments = []  # list of instrument objects
        self.acquisition_parameters = {
            'num_frames': 1,  # Common to all modalities
            'split_percentage': 50,  # Only used for split data stream modality
            'save_enabled': False,  # Whether to save acquired data
            'save_path': '',  # Full path to base filename for saving data
            
            # Galvo acquisition parameters (moved from galvo instrument)
            'dwell_time': 10,  # Per pixel dwell time in microseconds
            'extrasteps_left': 50,  # Extra steps left in fast direction
            'extrasteps_right': 50,  # Extra steps right in fast direction
            'amplitude_x': 0.5,  # Amplitude for X axis
            'amplitude_y': 0.5,  # Amplitude for Y axis
            'offset_x': 0.0,  # Offset for X axis
            'offset_y': 0.0,  # Offset for Y axis
            'x_pixels': 512,  # Number of X pixels for galvo scanning
            'y_pixels': 512,  # Number of Y pixels for galvo scanning
            
            # Prior stage acquisition parameters (moved from prior stage instrument)
            'numtiles_x': 10,  # Number of X tiles for acquisition
            'numtiles_y': 10,  # Number of Y tiles for acquisition  
            'numtiles_z': 5,   # Number of Z tiles for acquisition
            'tile_size_x': 100,  # Tile size in µm for X
            'tile_size_y': 100,  # Tile size in µm for Y
            'tile_size_z': 50,   # Tile size in µm for Z
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
        
        self.rpoc_masks = {}
        self.rpoc_channels = {}

    def get_instruments_by_type(self, instrument_type):
        '''wrapper for getting the instruments to verify if all the necessary instruments are connected in each modality'''
        return get_instruments_by_type(self.instruments, instrument_type)
    
    def serialize_instruments(self):
        '''serialize instrument data types for saving in json'''
        serialized = []
        for instrument in self.instruments:
            serialized.append({
                'name': instrument.name,
                'instrument_type': instrument.instrument_type,
                'parameters': instrument.parameters.copy()
            })
        return serialized
    
    def deserialize_instruments(self, serialized_instruments):
        '''undo serialize_instrument()'''
        self.instruments.clear()
        for data in serialized_instruments:
            try:
                # instrument names must be strings
                instrument_name = data.get('name', 'Unknown Instrument')
                
                # parameters must be a dict
                parameters = data.get('parameters', {})
                
                instrument = create_instrument(
                    data['instrument_type'], 
                    instrument_name, 
                    parameters
                )
                self.instruments.append(instrument)
            except Exception as e:
                print(f"Failed to recreate instrument {data.get('name', 'Unknown')}: {e}")

class StateSignalBus(QObject):
    '''
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
    instrument_removed = pyqtSignal(object) # emits when an instrument is removed (instrument_object)
    instrument_updated = pyqtSignal(object) # emits when an instrument is updated (instrument_object)

    console_message = pyqtSignal(str) # emits console status messages

    rpoc_enabled_changed = pyqtSignal(bool) # emits when RPOC enabled checkbox is toggled
    acquisition_parameter_changed = pyqtSignal(str, object) # emits when acquisition parameters change (param_name, new_value)
    save_path_changed = pyqtSignal(str) # emits when save path is changed
    
    # UI state signals
    ui_state_changed = pyqtSignal(str, object) # emits when UI state changes (param_name, new_value)
    lines_toggled = pyqtSignal(bool) # emits when lines are toggled
    
    # Unified data signal - handles both frame updates and final data
    data_signal = pyqtSignal(object, int, int, bool) # data, index, total, is_final

    # RPOC mask creation
    mask_created = pyqtSignal(object) # emits when a mask is created
    rpoc_channel_removed = pyqtSignal(int) # emits when an RPOC channel is removed (channel_id)

    def __init__(self):
        super().__init__()
        self.connected = False

    def disconnect_all(self):
        if self.connected:
            self.disconnect()
            self.connected = False

    def bind_controllers(self, app_state, main_window):
        # recalls when gui is rebuilt on modality changes
        # need to disconnect the old connections
        self.disconnect_all()
        
        # store reference to app_state for data saving
        self.app_state = app_state
        
        self.load_config_btn_clicked.connect(lambda: handle_load_config(app_state, main_window))
        self.save_config_btn_clicked.connect(lambda: handle_save_config(app_state))

        self.continuous_btn_clicked.connect(lambda: handle_continuous_acquisition(app_state, self))
        self.single_btn_clicked.connect(lambda: handle_single_acquisition(app_state, self))
        self.stop_btn_clicked.connect(lambda: handle_stop_acquisition(app_state, self))

        self.modality_dropdown_changed.connect(lambda text: handle_modality_changed(text, app_state, main_window))
        self.add_instrument_btn_clicked.connect(lambda: handle_add_instrument(app_state, main_window))
        self.add_modality_instrument.connect(lambda instrument_type: handle_add_modality_instrument(instrument_type, app_state, main_window))
        self.instrument_removed.connect(lambda instrument: handle_instrument_removed(instrument, app_state))
        self.instrument_updated.connect(lambda instrument: handle_instrument_updated(instrument, app_state, main_window))

        self.data_signal.connect(lambda data, idx, total, is_final: handle_data_signal(data, idx, total, is_final, app_state, main_window))
        self.console_message.connect(lambda message: handle_console_message(message, app_state, main_window))

        self.rpoc_enabled_changed.connect(lambda enabled: handle_rpoc_enabled_changed(enabled, app_state))
        self.acquisition_parameter_changed.connect(lambda param_name, value: handle_acquisition_parameter_changed(param_name, value, app_state, main_window))
        self.save_path_changed.connect(lambda path: handle_save_path_changed(path, app_state))
        
        self.ui_state_changed.connect(lambda param_name, value: handle_ui_state_changed(param_name, value, app_state))
        self.lines_toggled.connect(lambda enabled: handle_lines_toggled(enabled, app_state, main_window))
        
        self.mask_created.connect(lambda mask: handle_mask_created(mask, app_state, main_window, self))
        self.rpoc_channel_removed.connect(lambda channel_id: handle_rpoc_channel_removed(channel_id, app_state, self))

        # lines <-> Image Display signal wiring
        try:
            image_display = main_window.mid_layout.image_display_widget
            lines = main_window.mid_layout.lines_widget
            image_display.connect_lines_widget(lines)

        except Exception as e:
            print(f"Error connecting lines widget: {e}")

        # put remaining singla wiring between image displa widgets and dockable helper widgets
        
        self.connected = True





class AcquisitionWorker(QObject):
    acquisition_finished = pyqtSignal(object)

    def __init__(self, acquisition, continuous=False):
        super().__init__()
        self.acquisition = acquisition
        self.continuous = continuous
        self.running = True

    def run(self):
        while self.running:
            self.acquisition.set_stop_flag(lambda: not self.running)
            result = self.acquisition.perform_acquisition()
            
            if self.running: 
                self.acquisition_finished.emit(result)
            
            if not self.continuous:
                break

    def stop(self):
        self.running = False





def handle_load_config(app_state, main_window):
    '''load the config.json file and update the main_window per the config'''
    try:
        file_path, _ = QFileDialog.getOpenFileName(
            None, 'Load Configuration', '', 
            'JSON files (*.json);;All files (*)'
        )
        if not file_path:
            return 0
            
        with open(file_path, 'r') as f:
            config_data = json.load(f)
        
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
        
        if 'instruments' in config_data:
            app_state.deserialize_instruments(config_data['instruments'])
        
        # Handle rpoc_masks - clear actual mask data but preserve channel structure
        if 'rpoc_masks' in config_data:
            # Clear existing masks and set up structure for new ones
            app_state.rpoc_masks = {}
            # Note: Actual mask data is not loaded from config, users need to reload masks
        
        if 'rpoc_channels' in config_data:
            app_state.rpoc_channels = config_data['rpoc_channels']
        
        print(f"Configuration loaded from {file_path}")
        main_window.rebuild_gui()
        
        # Re-establish lines widget connections after GUI rebuild
        try:
            image_display = main_window.mid_layout.image_display_widget
            lines = main_window.mid_layout.lines_widget
            image_display.connect_lines_widget(lines)
            print("Lines widget connections re-established after config load")
        except Exception as e:
            print(f"Error re-establishing lines widget connections after config load: {e}")

        return 1
        
    except Exception as e:
        QMessageBox.critical(None, "Load Error", f"Failed to load configuration: {e}")
        return 0

def handle_save_config(app_state):
    '''save current app state into a config.json in a format for handle_load_config() to read later'''
    try:
        file_path, _ = QFileDialog.getSaveFileName(
            None, 'Save Configuration', 'config.json', 
            'JSON files (*.json);;All files (*)'
        )
        if not file_path:
            return 0
        
        # Create a serializable version of rpoc_masks that excludes the actual mask data
        serializable_rpoc_masks = {}
        if hasattr(app_state, 'rpoc_masks'):
            for channel_id, mask in app_state.rpoc_masks.items():
                if mask is not None:
                    # Store metadata about the mask instead of the actual mask data
                    serializable_rpoc_masks[str(channel_id)] = {
                        'shape': mask.shape if hasattr(mask, 'shape') else None,
                        'dtype': str(mask.dtype) if hasattr(mask, 'dtype') else None,
                        'has_mask': True
                    }
                else:
                    serializable_rpoc_masks[str(channel_id)] = {
                        'has_mask': False
                    }
        
        config_data = {
            'modality': app_state.modality,
            'acquisition_parameters': app_state.acquisition_parameters.copy(),
            'display_parameters': app_state.display_parameters.copy(),
            'rpoc_enabled': app_state.rpoc_enabled,
            'ui_state': app_state.ui_state.copy(),
            'instruments': app_state.serialize_instruments(),
            'rpoc_masks': serializable_rpoc_masks,
            'rpoc_channels': app_state.rpoc_channels.copy()
            # Note: current_data is intentionally excluded as it may contain numpy arrays
        }
        
        with open(file_path, 'w') as f:
            json.dump(config_data, f, indent=2)
        
        print(f"Configuration saved to {file_path}")
        return 1
        
    except Exception as e:
        QMessageBox.critical(None, "Save Error", f"Failed to save configuration: {e}")
        return 0

def handle_continuous_acquisition(app_state, signal_bus):
    return handle_single_acquisition(app_state, signal_bus, continuous=True)

def handle_single_acquisition(app_state, signal_bus, continuous=False):
    # take snapshot of all parameters rather than continually reading them from app_state during acquisition
    modality = app_state.modality
    parameters = app_state.acquisition_parameters.copy() 
    rpoc_enabled = app_state.rpoc_enabled

    if continuous:
        signal_bus.console_message.emit(f"Starting continuous {modality} acquisition...")
    else:
        signal_bus.console_message.emit(f"Starting {modality} acquisition...")
    try:
        validate_acquisition_parameters(parameters, modality)
    except ValueError as e:
        signal_bus.console_message.emit(f"Parameter validation error: {e}")
        return

    instruments = {}
    for instrument_type in ['galvo', 'data input', 'delay stage', 'prior stage']:
        instruments[instrument_type] = app_state.get_instruments_by_type(instrument_type)
    
    # check modality parameters (instruments in particular)
    if modality == 'confocal':
        galvo = instruments.get('galvo', []) # .get() returns [] instead of None here, which is easier to look at in if statement that follows
        data_inputs = instruments.get('data input', [])
        
        if not galvo: 
            signal_bus.console_message.emit("Error: Confocal acquisition requires at least one galvo instrument. Please add a galvo scanner.")
            return
        
        if not data_inputs:
            signal_bus.console_message.emit("Error: Confocal acquisition requires at least one data input instrument. Please add a data input.")
            return
    
    elif modality == 'split data stream':
        galvo = instruments.get('galvo', [])
        data_inputs = instruments.get('data input', [])
        prior_stage = instruments.get('prior stage', [])
        
        if not galvo:
            signal_bus.console_message.emit("Error: Split Data Stream acquisition requires at least one galvo instrument. Please add a galvo scanner.")
            return
        
        if not data_inputs:
            signal_bus.console_message.emit("Error: Split Data Stream acquisition requires at least one data input instrument. Please add a data input.")
            return
        
        if not prior_stage:
            signal_bus.console_message.emit("Error: Split Data Stream acquisition requires at least one prior stage instrument. Please add a prior stage.")
            return

    acquisition = None
    try:
        save_enabled = parameters.get('save_enabled', False)
        save_path = parameters.get('save_path', '')
        
        match modality:
            case 'simulated':
                acquisition = Simulated(signal_bus=signal_bus, 
                                      acquisition_parameters=parameters,
                                      save_enabled=save_enabled, save_path=save_path)
                
            case 'confocal':
                signal_bus.console_message.emit("Confocal acquisition started")

                # have already verified that the instruments exist, no need to .get() here
                galvo = instruments['galvo'][0] # returned as a list because there are multiple of each instrument in general
                data_inputs = instruments['data input']
                
                acquisition = Confocal(
                    galvo=galvo, 
                    data_inputs=data_inputs,
                    num_frames=parameters['num_frames'],
                    signal_bus=signal_bus,
                    acquisition_parameters=parameters,
                    save_enabled=save_enabled,
                    save_path=save_path
                )
                
                # configure RPOC with masks, channels, and static channel information
                if rpoc_enabled:
                    rpoc_masks = getattr(app_state, 'rpoc_masks', {})
                    rpoc_channels = getattr(app_state, 'rpoc_channels', {})
                    rpoc_static_channels = getattr(app_state, 'rpoc_static_channels', {})
                    signal_bus.console_message.emit(f"Acquisition RPOC - enabled: {rpoc_enabled}, masks: {len(rpoc_masks)}, channels: {len(rpoc_channels)}, static: {len(rpoc_static_channels)}")
                    acquisition.configure_rpoc(rpoc_enabled, rpoc_masks=rpoc_masks, rpoc_channels=rpoc_channels, rpoc_static_channels=rpoc_static_channels)
                else:
                    signal_bus.console_message.emit(f"Acquisition RPOC - disabled")
                
            case 'split data stream':
                signal_bus.console_message.emit("Split Data Stream acquisition started")

                # have already verified that the instruments exist, no need to .get() here
                galvo = instruments['galvo'][0] # returned as a list because there are multiple of each instrument in general
                data_inputs = instruments['data input']
                prior_stage = instruments['prior stage'][0] if 'prior stage' in instruments else None
                split_percentage = parameters.get('split_percentage', 50)
                
                acquisition = SplitDataStream(
                    galvo=galvo, 
                    data_inputs=data_inputs,
                    prior_stage=prior_stage,
                    num_frames=parameters['num_frames'],
                    split_percentage=split_percentage,
                    signal_bus=signal_bus,
                    acquisition_parameters=parameters,
                    save_enabled=save_enabled,
                    save_path=save_path
                )
                
                # configure RPOC with masks, channels, and static channel information
                if rpoc_enabled:
                    rpoc_masks = getattr(app_state, 'rpoc_masks', {})
                    rpoc_channels = getattr(app_state, 'rpoc_channels', {})
                    rpoc_static_channels = getattr(app_state, 'rpoc_static_channels', {})
                    signal_bus.console_message.emit(f"Acquisition RPOC - enabled: {rpoc_enabled}, masks: {len(rpoc_masks)}, channels: {len(rpoc_channels)}, static: {len(rpoc_static_channels)}")
                    acquisition.configure_rpoc(rpoc_enabled, rpoc_masks=rpoc_masks, rpoc_channels=rpoc_channels, rpoc_static_channels=rpoc_static_channels)
                else:
                    signal_bus.console_message.emit(f"Acquisition RPOC - disabled")
                
            case _:
                signal_bus.console_message.emit('Warning: invalid modality, defaulting to simulation')
                default_params = {'x_pixels': 512, 'y_pixels': 512, 'num_frames': 1}
                acquisition = Simulated(signal_bus=signal_bus,
                                      acquisition_parameters=default_params,
                                      save_enabled=save_enabled, save_path=save_path)

    except Exception as e:
        signal_bus.console_message.emit(f'Error creating acquisition object: {e}')
        return

    if acquisition is not None:
        # acquisition objects receive parameters through their constructors and instruments are passed as needed
        # the clean parameter snapshot has already been validated and passed to the acquisition constructor.
        
        worker = AcquisitionWorker(acquisition, continuous=continuous)
        thread = QThread()
        worker.moveToThread(thread)

        acquisition.worker = worker
        worker.acquisition_finished.connect(lambda data: handle_acquisition_thread_finished(data, signal_bus, thread, worker))
        thread.started.connect(worker.run)
        thread.start()

        # garbage collection
        signal_bus.acq_thread = thread
        signal_bus.acq_worker = worker
    else:
        signal_bus.console_message.emit("Error: Failed to create acquisition object")

def validate_acquisition_parameters(parameters, modality):
    # All modalities now use acquisition parameters for pixel dimensions
    required_params = {
        'simulated': ['x_pixels', 'y_pixels', 'num_frames'],
        'confocal': [
            'x_pixels', 'y_pixels', 'num_frames',
            'dwell_time', 'extrasteps_left', 'extrasteps_right',
            'amplitude_x', 'amplitude_y', 'offset_x', 'offset_y'
        ],
        'split data stream': [
            'x_pixels', 'y_pixels', 'num_frames', 'split_percentage',
            'dwell_time', 'extrasteps_left', 'extrasteps_right',
            'amplitude_x', 'amplitude_y', 'offset_x', 'offset_y',
            'numtiles_x', 'numtiles_y', 'numtiles_z',
            'tile_size_x', 'tile_size_y', 'tile_size_z'
        ],
        'custom': ['x_pixels', 'y_pixels', 'num_frames']
    }
    
    missing_params = []
    for param in required_params[modality]:
        if param not in parameters:
            missing_params.append(param)
    
    if missing_params:
        raise ValueError(f"Missing required parameters for {modality}: {missing_params}")
    
    # Validate parameter ranges for all parameters that are present
    validation_rules = {
        'x_pixels': (1, 10000, "x_pixels must be between 1 and 10000"),
        'y_pixels': (1, 10000, "y_pixels must be between 1 and 10000"),
        'num_frames': (1, 10000, "num_frames must be between 1 and 10000"),
        'split_percentage': (1, 99, "split_percentage must be between 1 and 99"),
        'dwell_time': (1, 1000, "dwell_time must be between 1 and 1000 µs"),
        'extrasteps_left': (0, 10000, "extrasteps_left must be between 0 and 10000"),
        'extrasteps_right': (0, 10000, "extrasteps_right must be between 0 and 10000"),
        'amplitude_x': (0.01, 10.0, "amplitude_x must be between 0.01V and 10V"),
        'amplitude_y': (0.01, 10.0, "amplitude_y must be between 0.01V and 10V"),
        'offset_x': (-10.0, 10.0, "offset_x must be between -10V and 10V"),
        'offset_y': (-10.0, 10.0, "offset_y must be between -10V and 10V"),
        'numtiles_x': (1, 1000, "numtiles_x must be between 1 and 1000"),
        'numtiles_y': (1, 1000, "numtiles_y must be between 1 and 1000"),
        'numtiles_z': (1, 1000, "numtiles_z must be between 1 and 1000"),
        'tile_size_x': (-10000, 10000, "tile_size_x must be between -10000µm and 10000µm"),
        'tile_size_y': (-10000, 10000, "tile_size_y must be between -10000µm and 10000µm"),
        'tile_size_z': (-10000, 10000, "tile_size_z must be between -10000µm and 10000µm")
    }
    
    # Validate each parameter that is present in the parameters dict
    for param_name, param_value in parameters.items():
        if param_name in validation_rules:
            min_val, max_val, error_msg = validation_rules[param_name]
            if param_value < min_val or param_value > max_val:
                raise ValueError(error_msg)

def handle_acquisition_thread_finished(data, signal_bus, thread, worker):
    # for continuous acquisition, don't clean up the thread - let it continue
    if worker.continuous:
        return
    
    if hasattr(signal_bus, 'acq_thread'):
        thread = signal_bus.acq_thread
        worker = signal_bus.acq_worker
        worker.stop()
        thread.quit()
        thread.wait()
        worker.deleteLater()
        thread.deleteLater()
        del signal_bus.acq_thread
        del signal_bus.acq_worker
    signal_bus.console_message.emit("Acquisition complete!")
    # final data signal is now emitted by the acquisition classes themselves

def handle_stop_acquisition(app_state, signal_bus):
    if hasattr(signal_bus, 'acq_thread') and hasattr(signal_bus, 'acq_worker'):
        thread = signal_bus.acq_thread
        worker = signal_bus.acq_worker

        worker.stop()
        thread.quit()
        if thread.wait(5000):  # Wait up to 5 seconds
            thread.deleteLater()
            worker.deleteLater()
        else:
            thread.terminate()
            thread.wait()
            thread.deleteLater()
            worker.deleteLater()

        del signal_bus.acq_thread
        del signal_bus.acq_worker

        if hasattr(worker, 'continuous') and worker.continuous:
            signal_bus.console_message.emit("Continuous acquisition stopped")
        else:
            signal_bus.console_message.emit("Acquisition stopped")
    return 0

def handle_console_message(message, app_state, main_window):
    main_window.top_bar.add_console_message(message)
    return 0

def handle_modality_changed(text, app_state, main_window):
    app_state.modality = text.lower()

    handle_stop_acquisition(app_state,main_window.signals)

    app_state.current_data = None

    main_window.rebuild_gui()
    
    main_window.signals.bind_controllers(app_state, main_window)
    
    return 0



def handle_add_instrument(app_state, main_window):
    instrument_type = show_add_instrument_dialog(main_window)
    if instrument_type:
        handle_add_modality_instrument(instrument_type, app_state, main_window)
    return 0

def handle_add_modality_instrument(instrument_type, app_state, main_window):
    # Use the instrument manager to show configuration dialog
    instrument, display_name = show_configure_instrument_dialog(
        instrument_type, 
        main_window, 
        main_window.signals.console_message.emit
    )
    
    if instrument:
        if instrument.initialize():
            if not hasattr(app_state, 'instruments'):
                app_state.instruments = []
            
            app_state.instruments.append(instrument)

            main_window.left_widget.instrument_controls.add_instrument(instrument)
            main_window.left_widget.instrument_controls.rebuild()
            main_window.signals.console_message.emit(f"Added {display_name} successfully")
        else:
            main_window.signals.console_message.emit(f"Failed to connect to {display_name}")
    
    return 0

def handle_instrument_removed(instrument, app_state):
    if hasattr(app_state, 'instruments') and instrument in app_state.instruments:
        # Disconnect the instrument before removing it
        try:
            instrument.disconnect()
        except Exception as e:
            print(f"Error disconnecting instrument {instrument.name}: {e}")
        
        app_state.instruments.remove(instrument)
        # Update modality buttons to show the button for the removed instrument type
        # This will be handled by the GUI when it rebuilds
    return 0

def handle_instrument_updated(instrument, app_state, main_window):
    """Handle instrument parameter updates"""
    print(f"Updated instrument: {instrument.name}")
    # Refresh channel labels in multichannel display if it exists
    if hasattr(main_window, 'mid_layout') and hasattr(main_window.mid_layout, 'image_display_widget'):
        display_widget = main_window.mid_layout.image_display_widget
        if hasattr(display_widget, 'refresh_channel_labels'):
            display_widget.update_channel_names()
            display_widget.refresh_channel_labels()
    return 0

def handle_rpoc_enabled_changed(enabled, app_state):
    app_state.rpoc_enabled = enabled
    print(f"RPOC enabled changed to: {enabled}")
    return 0

def handle_acquisition_parameter_changed(param_name, value, app_state, main_window=None):
    if param_name in app_state.acquisition_parameters:
        app_state.acquisition_parameters[param_name] = value
    return 0

def handle_save_path_changed(path, app_state):
    app_state.acquisition_parameters['save_path'] = path  # Full path to base filename for saving data
    return 0



def handle_ui_state_changed(param_name, value, app_state):
    if param_name in app_state.ui_state:
        app_state.ui_state[param_name] = value
    return 0

def handle_lines_toggled(enabled, app_state, main_window=None):
    app_state.ui_state['lines_enabled'] = enabled
    main_window.mid_layout.on_lines_toggled(enabled)
    
    return 0

def handle_data_signal(data, idx, total, is_final, app_state, main_window):
    # route the unified data signal to the widget
    if hasattr(main_window, 'mid_layout') and hasattr(main_window.mid_layout, 'image_display_widget'):
        widget = main_window.mid_layout.image_display_widget
        if hasattr(widget, 'handle_data_signal'):
            widget.handle_data_signal(data, idx, total, is_final)
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
    
    # remove the DAQ channel info from app_state if it exists
    if hasattr(app_state, 'rpoc_channels') and channel_id in app_state.rpoc_channels:
        del app_state.rpoc_channels[channel_id]
    
    signal_bus.console_message.emit(f"RPOC channel {channel_id} removed.")
    return 0