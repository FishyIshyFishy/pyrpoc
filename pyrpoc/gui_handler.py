import json
from PyQt6.QtWidgets import QFileDialog, QMessageBox, QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QComboBox, QLabel
from pyrpoc.instruments.instrument_manager import create_instrument, get_instruments_by_type
from PyQt6.QtCore import QObject, pyqtSignal, QThread
from pyrpoc.acquisitions import *
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
            'x_pixels': 512,  # Only used for non-confocal modalities (confocal uses galvo parameters)
            'y_pixels': 512,  # Only used for non-confocal modalities (confocal uses galvo parameters)
            'split_percentage': 50,  # Only used for split data stream modality
            'save_enabled': False,  # Whether to save acquired data
            'save_path': '',  # File path for saving data
            
            # Galvo acquisition parameters (moved from galvo instrument)
            'dwell_time': 10e-6,  # Per pixel dwell time in seconds
            'extrasteps_left': 50,  # Extra steps left in fast direction
            'extrasteps_right': 50,  # Extra steps right in fast direction
            'amplitude_x': 0.5,  # Amplitude for X axis
            'amplitude_y': 0.5,  # Amplitude for Y axis
            'offset_x': 0.0,  # Offset for X axis
            'offset_y': 0.0,  # Offset for Y axis
            
            # Prior stage acquisition parameters (moved from prior stage instrument)
            'numsteps_x': 10,  # Number of X steps for acquisition
            'numsteps_y': 10,  # Number of Y steps for acquisition  
            'numsteps_z': 5,   # Number of Z steps for acquisition
            'step_size_x': 100,  # Step size in µm for X
            'step_size_y': 100,  # Step size in µm for Y
            'step_size_z': 50,   # Step size in µm for Z
            'max_z_height': 50000,  # Maximum Z height in µm
            'safe_move_distance': 10000  # Safe movement distance in µm
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
        
        if 'rpoc_masks' in config_data:
            app_state.rpoc_masks = config_data['rpoc_masks']
        if 'rpoc_channels' in config_data:
            app_state.rpoc_channels = config_data['rpoc_channels']
        
        print(f"Configuration loaded from {file_path}")
        main_window.rebuild_gui()

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
        
        config_data = {
            'modality': app_state.modality,
            'acquisition_parameters': app_state.acquisition_parameters.copy(),
            'display_parameters': app_state.display_parameters.copy(),
            'rpoc_enabled': app_state.rpoc_enabled,
            'ui_state': app_state.ui_state.copy(),
            'instruments': app_state.serialize_instruments(),
            'rpoc_masks': app_state.rpoc_masks.copy(),
            'rpoc_channels': app_state.rpoc_channels.copy()
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
    for instrument_type in ['Galvo', 'Data Input', 'Delay Stage', 'Zaber Stage', 'Prior Stage']:
        instruments[instrument_type] = app_state.get_instruments_by_type(instrument_type)
    
    # check modality parameters (instruments in particular)
    if modality == 'confocal':
        galvo = instruments.get('Galvo', []) # .get() returns [] instead of None here, which is easier to look at in if statement that follows
        data_inputs = instruments.get('Data Input', [])
        
        if not galvo: 
            signal_bus.console_message.emit("Error: Confocal acquisition requires at least one Galvo instrument. Please add a Galvo scanner.")
            return
        
        if not data_inputs:
            signal_bus.console_message.emit("Error: Confocal acquisition requires at least one Data Input instrument. Please add a Data Input.")
            return
    
    elif modality == 'widefield':
        data_inputs = instruments.get('Data Input', [])
        
        if not data_inputs:
            signal_bus.console_message.emit("Error: Widefield acquisition requires at least one Data Input instrument. Please add a Data Input.")
            return

    elif modality == 'split data stream':
        galvo = instruments.get('Galvo', [])
        data_inputs = instruments.get('Data Input', [])
        prior_stage = instruments.get('Prior Stage', [])
        
        if not galvo:
            signal_bus.console_message.emit("Error: Split Data Stream acquisition requires at least one Galvo instrument. Please add a Galvo scanner.")
            return
        
        if not data_inputs:
            signal_bus.console_message.emit("Error: Split Data Stream acquisition requires at least one Data Input instrument. Please add a Data Input.")
            return
        
        if not prior_stage:
            signal_bus.console_message.emit("Error: Split Data Stream acquisition requires at least one Prior Stage instrument. Please add a Prior Stage.")
            return

    elif modality == 'mosaic':
        galvo = instruments.get('Galvo', [])
        data_inputs = instruments.get('Data Input', [])
        
        if not galvo:
            signal_bus.console_message.emit("Error: Mosaic acquisition requires at least one Galvo instrument. Please add a Galvo scanner.")
            return
        
        if not data_inputs:
            signal_bus.console_message.emit("Error: Mosaic acquisition requires at least one Data Input instrument. Please add a Data Input.")
            return

    acquisition = None
    try:
        save_enabled = parameters.get('save_enabled', False)
        save_path = parameters.get('save_path', '')
        
        match modality:
            case 'simulated':
                x_pixels = parameters['x_pixels']
                y_pixels = parameters['y_pixels']
                num_frames = parameters['num_frames']
                acquisition = Simulated(x_pixels, y_pixels, num_frames, signal_bus, 
                                      save_enabled=save_enabled, save_path=save_path)
                
            case 'widefield':
                x_pixels = parameters['x_pixels']
                y_pixels = parameters['y_pixels']

                data_inputs = instruments.get('Data Input', [])
                signal_bus.console_message.emit(f"Widefield acquisition: {x_pixels}x{y_pixels}")
                acquisition = Widefield(data_inputs=data_inputs, signal_bus=signal_bus,
                                      save_enabled=save_enabled, save_path=save_path)
                
            case 'confocal':
                signal_bus.console_message.emit("Confocal acquisition started")

                # have already verified that the instruments exist, no need to .get() here
                galvo = instruments['Galvo'][0] # returned as a list because there are multiple of each instrument in general
                data_inputs = instruments['Data Input']
                
                acquisition = Confocal(
                    galvo=galvo, 
                    data_inputs=data_inputs,
                    num_frames=parameters['num_frames'],
                    signal_bus=signal_bus,
                    acquisition_parameters=parameters,
                    save_enabled=save_enabled,
                    save_path=save_path
                )
                
                # configure RPOC with both masks and channel information
                if rpoc_enabled:
                    rpoc_masks = getattr(app_state, 'rpoc_masks', {})
                    rpoc_channels = getattr(app_state, 'rpoc_channels', {})
                    signal_bus.console_message.emit(f"Acquisition RPOC - enabled: {rpoc_enabled}, masks: {len(rpoc_masks)}, channels: {len(rpoc_channels)}")
                    acquisition.configure_rpoc(rpoc_enabled, rpoc_masks=rpoc_masks, rpoc_channels=rpoc_channels)
                else:
                    signal_bus.console_message.emit(f"Acquisition RPOC - disabled")
                
            case 'split data stream':
                signal_bus.console_message.emit("Split Data Stream acquisition started")

                # have already verified that the instruments exist, no need to .get() here
                galvo = instruments['Galvo'][0] # returned as a list because there are multiple of each instrument in general
                data_inputs = instruments['Data Input']
                prior_stage = instruments['Prior Stage'][0] if 'Prior Stage' in instruments else None
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
                
                # configure RPOC with both masks and channel information
                if rpoc_enabled:
                    rpoc_masks = getattr(app_state, 'rpoc_masks', {})
                    rpoc_channels = getattr(app_state, 'rpoc_channels', {})
                    signal_bus.console_message.emit(f"Acquisition RPOC - enabled: {rpoc_enabled}, masks: {len(rpoc_masks)}, channels: {len(rpoc_channels)}")
                    acquisition.configure_rpoc(rpoc_enabled, rpoc_masks=rpoc_masks, rpoc_channels=rpoc_channels)
                else:
                    signal_bus.console_message.emit(f"Acquisition RPOC - disabled")
                
            case 'mosaic':
                signal_bus.console_message.emit("Mosaic acquisition started")
                acquisition = Mosaic(signal_bus=signal_bus,
                                   save_enabled=save_enabled, save_path=save_path)
                
            case 'zscan':
                signal_bus.console_message.emit("ZScan acquisition started")
                acquisition = ZScan(signal_bus=signal_bus,
                                  save_enabled=save_enabled, save_path=save_path)
                
            case 'custom':
                signal_bus.console_message.emit("Custom acquisition started")
                acquisition = Custom(signal_bus=signal_bus,
                                   save_enabled=save_enabled, save_path=save_path)
                
            case _:
                signal_bus.console_message.emit('Warning: invalid modality, defaulting to simulation')
                acquisition = Simulated(512, 512, 1, signal_bus,
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
        'widefield': ['x_pixels', 'y_pixels', 'num_frames'],
        'confocal': ['x_pixels', 'y_pixels', 'num_frames'],  # Now uses acquisition parameters
        'split data stream': ['x_pixels', 'y_pixels', 'num_frames', 'split_percentage'],  # Now uses acquisition parameters
        'mosaic': ['x_pixels', 'y_pixels', 'num_frames'],
        'zscan': ['x_pixels', 'y_pixels', 'num_frames'],
        'custom': ['x_pixels', 'y_pixels', 'num_frames']
    }
    
    missing_params = []
    for param in required_params[modality]:
        if param not in parameters:
            missing_params.append(param)
    
    if missing_params:
        raise ValueError(f"Missing required parameters for {modality}: {missing_params}")
    
    if 'x_pixels' in parameters and (parameters['x_pixels'] <= 0 or parameters['x_pixels'] > 10000):
        raise ValueError("x_pixels must be between 1 and 10000")
    
    if 'y_pixels' in parameters and (parameters['y_pixels'] <= 0 or parameters['y_pixels'] > 10000):
        raise ValueError("y_pixels must be between 1 and 10000")
    
    if 'num_frames' in parameters and (parameters['num_frames'] <= 0 or parameters['num_frames'] > 10000):
        raise ValueError("num_frames must be between 1 and 10000")
    
    if 'split_percentage' in parameters and (parameters['split_percentage'] <= 0 or parameters['split_percentage'] >= 100):
        raise ValueError("split_percentage must be between 1 and 99")

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
    dialog = QDialog(main_window)
    dialog.setWindowTitle("Add Instrument")
    dialog.setModal(True)
    
    layout = QVBoxLayout()
    
    layout.addWidget(QLabel("Select instrument type:"))
    
    combo = QComboBox()
    combo.addItems(['Delay Stage', 'Zaber Stage', 'Prior Stage'])
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
    # Create instrument with default parameters
    instrument = create_instrument(instrument_type, instrument_type)
    
    # Get the unified widget for configuration
    unified_widget = instrument.get_widget()
    if unified_widget:
        dialog = QDialog(main_window)
        dialog.setWindowTitle(f"Configure {instrument_type}")
        dialog.setModal(True)
        
        layout = QVBoxLayout()
        layout.addWidget(unified_widget)
        
        # Buttons
        button_layout = QHBoxLayout()
        ok_btn = QPushButton("OK")
        cancel_btn = QPushButton("Cancel")
        
        ok_btn.clicked.connect(dialog.accept)
        cancel_btn.clicked.connect(dialog.reject)
        
        button_layout.addWidget(ok_btn)
        button_layout.addWidget(cancel_btn)
        layout.addLayout(button_layout)
        
        dialog.setLayout(layout)
        dialog.resize(400, 300)
        
        if dialog.exec() == QDialog.DialogCode.Accepted:
            parameters = unified_widget.get_parameters()
            
            if parameters is not None:  
                instrument_name = parameters.get('name', f'{instrument_type}')
                instrument = create_instrument(instrument_type, instrument_name, parameters)
                
                if hasattr(instrument, 'name') and instrument.name:
                    display_name = instrument.name
                else:
                    display_name = instrument_name
            else:
                main_window.signals.console_message.emit(f"Failed to create {instrument_type} - invalid parameters")
                return 0
            
            if instrument.initialize():
                if not hasattr(app_state, 'instruments'):
                    app_state.instruments = []
                
                app_state.instruments.append(instrument)

                main_window.left_widget.instrument_controls.add_instrument(instrument)
                main_window.left_widget.instrument_controls.rebuild()
                main_window.signals.console_message.emit(f"Added {display_name} successfully")
            else:
                main_window.signals.console_message.emit(f"Failed to connect to {display_name}")
    else:
        main_window.signals.console_message.emit(f"Failed to get configuration widget for {instrument_type}")
    
    return 0

def handle_instrument_removed(instrument, app_state):
    if hasattr(app_state, 'instruments') and instrument in app_state.instruments:
        app_state.instruments.remove(instrument)
        # Update modality buttons to show the button for the removed instrument type
        # This will be handled by the GUI when it rebuilds
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
    app_state.acquisition_parameters['save_path'] = path
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