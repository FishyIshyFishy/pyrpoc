import json
from PyQt6.QtWidgets import QFileDialog, QMessageBox
from pyrpoc.instruments.instrument_manager import create_instrument, get_instruments_by_type, show_add_instrument_dialog, show_configure_instrument_dialog
from pyrpoc.modalities import modality_registry
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
            'aom_delay': 0,  # AOM delay in microseconds for split data stream modality
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
            
            # Fish modality parameters
            'offset_drift_x': 0.0,  # Drift offset for X axis in V
            'offset_drift_y': 0.0,  # Drift offset for Y axis in V
            'repetitions': 1,  # Number of treatment repetitions
            'ttl_device': 'Dev1',  # TTL device for local RPOC
            'ttl_port_line': 'port0/line0',  # TTL port/line for local RPOC
            'pfi_line': 'None',  # PFI line for timing (optional)
            'local_extrasteps_left': 50,  # Local RPOC extra steps left
            'local_extrasteps_right': 50,  # Local RPOC extra steps right
            'local_rpoc_dwell_time': 10,  # Local RPOC dwell time in microseconds
        }
        self.display_parameters = {
            # overlay parameter removed - not implemented yet
        }
        self.current_data = None
        
        # Display selection - stores the class name of the selected display
        self.selected_display = 'ImageDisplayWidget'  # Default to single channel display
        
        # UI state parameters
        self.ui_state = {
            'acquisition_parameters_visible': True,
            'instrument_controls_visible': True,
            'display_controls_visible': True,
            'main_splitter_sizes': [200, 800, 200],  # left, middle, right panel sizes
            'vertical_splitter_sizes': [100, 800],  # top bar, main content sizes
        }
        
        # RPOC storage - separate by channel type for clarity
        self.rpoc_mask_channels = {}    # channel_id -> {device, port_line, mask_data}
        self.rpoc_static_channels = {}  # channel_id -> {device, port_line, level}
        self.rpoc_script_channels = {}  # channel_id -> {device, port_line, script_data} (for future use)

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

    acquisition_parameter_changed = pyqtSignal(str, object) # emits when acquisition parameters change (param_name, new_value)
    save_path_changed = pyqtSignal(str) # emits when save path is changed
    
    # UI state signals
    ui_state_changed = pyqtSignal(str, object) # emits when UI state changes (param_name, new_value)
    
    # New acquisition pipeline signals
    display_setup = pyqtSignal(str) # display_class_name - requests display setup for acquisition
    acquisition_started = pyqtSignal() # emits when acquisition starts
    acquisition_stopped = pyqtSignal() # emits when acquisition stops
    data_received = pyqtSignal(object) # data - individual data frame during acquisition
    acquisition_complete = pyqtSignal() # signals that acquisition is complete
    
    mask_created = pyqtSignal(object) # emits when a mask is created
    rpoc_channel_removed = pyqtSignal(int) # emits when an RPOC channel is removed (channel_id)
    

    local_rpoc_started = pyqtSignal(object) # emits when local RPOC treatment is started (parameters)
    local_rpoc_progress = pyqtSignal(int) # emits progress updates (repetition_number)
    
    

    def __init__(self):
        super().__init__()
        self.connected = False

    def disconnect_all(self):
        if self.connected:
            self.disconnect()
            self.connected = False

    def bind_controllers(self, app_state, main_window):
        self.disconnect_all()

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

        # New acquisition pipeline signal handlers
        self.display_setup.connect(lambda display_class: handle_display_setup(display_class, app_state, main_window))
        self.data_received.connect(lambda data: handle_data_received(data, app_state, main_window))
        self.acquisition_complete.connect(lambda: handle_acquisition_complete(app_state, main_window))
        
        self.console_message.connect(lambda message: handle_console_message(message, app_state, main_window))


        
        # connect button state management signals
        self.acquisition_started.connect(lambda: handle_acquisition_started(main_window))
        self.acquisition_stopped.connect(lambda: handle_acquisition_stopped(main_window))
        
        # initialize button states
        main_window.top_bar.on_acquisition_stopped()
        
        self.acquisition_parameter_changed.connect(lambda param_name, value: handle_acquisition_parameter_changed(param_name, value, app_state, main_window))
        self.save_path_changed.connect(lambda path: handle_save_path_changed(path, app_state))
        
        self.ui_state_changed.connect(lambda param_name, value: handle_ui_state_changed(param_name, value, app_state))
        
        self.mask_created.connect(lambda mask: handle_mask_created(mask, app_state, main_window, self))
        self.rpoc_channel_removed.connect(lambda channel_id: handle_rpoc_channel_removed(channel_id, app_state, self))
        self.local_rpoc_started.connect(lambda parameters: handle_local_rpoc_started(parameters, app_state, self))
        self.local_rpoc_progress.connect(lambda repetition: handle_local_rpoc_progress(repetition, app_state, self))

        self.connected = True


def handle_acquisition_started(main_window):
    main_window.top_bar.on_acquisition_started()
    # display widget call for handle_display_setup here

def handle_acquisition_stopped(main_window):
    main_window.top_bar.on_acquisition_stopped()


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
        if 'ui_state' in config_data:
            app_state.ui_state.update(config_data['ui_state'])
        
        if 'instruments' in config_data:
            app_state.deserialize_instruments(config_data['instruments'])
        
            app_state.rpoc_mask_channels = {}
            for channel_id_str, channel_data in config_data['rpoc_mask_channels'].items():
                try:
                    channel_id = int(channel_id_str)
                    if 'enabled' not in channel_data:
                        channel_data['enabled'] = True
                    app_state.rpoc_mask_channels[channel_id] = channel_data
                except ValueError:
                    if 'enabled' not in channel_data:
                        channel_data['enabled'] = True
                    app_state.rpoc_mask_channels[channel_id_str] = channel_data

        elif 'rpoc_masks' in config_data:
            app_state.rpoc_mask_channels = {}
        
        if 'rpoc_script_channels' in config_data:
            app_state.rpoc_script_channels = {}
            for channel_id_str, channel_data in config_data['rpoc_script_channels'].items():
                try:
                    channel_id = int(channel_id_str)
                    if 'enabled' not in channel_data:
                        channel_data['enabled'] = True
                    app_state.rpoc_script_channels[channel_id] = channel_data
                except ValueError:
                    if 'enabled' not in channel_data:
                        channel_data['enabled'] = True
                    app_state.rpoc_script_channels[channel_id_str] = channel_data

        elif 'rpoc_channels' in config_data:
            app_state.rpoc_script_channels = {}
            for channel_id_str, channel_data in config_data['rpoc_channels'].items():
                try:
                    channel_id = int(channel_id_str)
                    if 'enabled' not in channel_data:
                        channel_data['enabled'] = True
                    app_state.rpoc_script_channels[channel_id] = channel_data
                except ValueError:
                    if 'enabled' not in channel_data:
                        channel_data['enabled'] = True
                    app_state.rpoc_script_channels[channel_id_str] = channel_data
        
        if 'rpoc_static_channels' in config_data:
            app_state.rpoc_static_channels = {}
            for channel_id_str, static_data in config_data['rpoc_static_channels'].items():
                try:
                    channel_id = int(channel_id_str)
                    if 'enabled' not in static_data:
                        static_data['enabled'] = True
                    app_state.rpoc_static_channels[channel_id] = static_data
                except ValueError:
                    if 'enabled' not in static_data:
                        static_data['enabled'] = True
                    app_state.rpoc_static_channels[channel_id_str] = static_data

        main_window.build_gui()
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
        
        # Create a serializable version of rpoc_mask_channels that excludes the actual mask data
        serializable_rpoc_mask_channels = {}
        if hasattr(app_state, 'rpoc_mask_channels'):
            for channel_id, channel_data in app_state.rpoc_mask_channels.items():
                if 'mask_data' in channel_data and channel_data['mask_data'] is not None:
                    # Store metadata about the mask instead of the actual mask data
                    serializable_rpoc_mask_channels[str(channel_id)] = {
                        'device': channel_data.get('device', 'Dev1'),
                        'port_line': channel_data.get('port_line', f'port0/line{4+channel_id-1}'),
                        'enabled': channel_data.get('enabled', True),
                        'mask_metadata': {
                            'shape': channel_data['mask_data'].shape if hasattr(channel_data['mask_data'], 'shape') else None,
                            'dtype': str(channel_data['mask_data'].dtype) if hasattr(channel_data['mask_data'], 'dtype') else None,
                            'has_mask': True
                        }
                    }
                else:
                    serializable_rpoc_mask_channels[str(channel_id)] = {
                        'device': channel_data.get('device', 'Dev1'),
                        'port_line': channel_data.get('port_line', f'port0/line{4+channel_id-1}'),
                        'enabled': channel_data.get('enabled', True),
                        'mask_metadata': {
                            'has_mask': False
                        }
                    }
        
        config_data = {
            'modality': app_state.modality,
            'acquisition_parameters': app_state.acquisition_parameters.copy(),
            'display_parameters': app_state.display_parameters.copy(),
            'ui_state': app_state.ui_state.copy(),
            'instruments': app_state.serialize_instruments(),
            'rpoc_mask_channels': serializable_rpoc_mask_channels,
            'rpoc_script_channels': app_state.rpoc_script_channels.copy(),
            'rpoc_static_channels': app_state.rpoc_static_channels.copy()
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
    modality_key = app_state.modality
    modality = modality_registry.get_modality(modality_key)
    parameters = app_state.acquisition_parameters.copy()
    instruments = app_state.instruments # no .copy() here?
    # rpoc_stuff = app_state.rpoc_stuff.copy()
    
    if not modality.validate_instruments(instruments):
        missing = [req for req in modality.required_instruments 
                  if not any(inst.instrument_type == req for inst in instruments)]
        signal_bus.console_message.emit(f"Error: {modality.name} acquisition requires instruments: {', '.join(missing)}")
        return
    instruments_by_type = {}
    for instrument_type in modality.required_instruments:
        instruments_by_type[instrument_type] = [inst for inst in instruments if inst.instrument_type == instrument_type]
    # check connection here?
    
    if not modality.validate_parameters(parameters):
        missing = [req for req in modality.required_parameters.keys() 
                  if req not in parameters]
        signal_bus.console_message.emit(f"Error: Missing required parameters for {modality.name}: {', '.join(missing)}")
        return
    
    # if not modality.validate_rpoc(rpoc_stuff)
        # issues = rpoc incompatibilities
        # signal_bus.emit(here are the incompatibilities)

    try:
        acquisition = modality.acquisition_class(
        signal_bus=signal_bus,
        parameters=parameters,
        **instruments_by_type
        )

        # ideally i can just write acquisition.configre_rpoc()        
        rpoc_mask_channels = getattr(app_state, 'rpoc_mask_channels', {})
        rpoc_static_channels = getattr(app_state, 'rpoc_static_channels', {})
        rpoc_script_channels = getattr(app_state, 'rpoc_script_channels', {})
        
        enabled_mask_channels = {k: v for k, v in rpoc_mask_channels.items() if v.get('enabled', True)}
        enabled_static_channels = {k: v for k, v in rpoc_static_channels.items() if v.get('enabled', True)}
        enabled_script_channels = {k: v for k, v in rpoc_script_channels.items() if v.get('enabled', True)}
        
        if len(enabled_mask_channels) + len(enabled_static_channels) + len(enabled_script_channels) > 0:
            acquisition.configure_rpoc(True, 
                                        rpoc_mask_channels=enabled_mask_channels, 
                                        rpoc_static_channels=enabled_static_channels, 
                                        rpoc_script_channels=enabled_script_channels)
    
    except Exception as e:
        signal_bus.console_message.emit(f'Error creating acquisition object: {e}')
        return

    try: 
        # need to move this into its own function for modularity
        signal_bus.display_setup.emit(app_state.selected_display)
        
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
        
        # emit signal to update button states
        signal_bus.acquisition_started.emit()
    
    except:
        signal_bus.console_message.emit("Error: Failed to create acquisition object")

def setup_acquisition_rpoc(app_state):
    return 0 # need to set this up but i dont know what the best way to do this is

def handle_acquisition_thread_finished(data, signal_bus, thread, worker):
    # for continuous acquisition, don't clean up the thread - let it continue
    if worker.continuous:
        return
    
    # clean up the thread and worker for non-continuous acquisitions
    worker.stop()
    thread.quit()
    thread.wait()
    worker.deleteLater()
    thread.deleteLater()
    
    # remove references from signal_bus
    if hasattr(signal_bus, 'acq_thread'):
        del signal_bus.acq_thread
    if hasattr(signal_bus, 'acq_worker'):
        del signal_bus.acq_worker
    
    # Legacy completion handling
    signal_bus.console_message.emit("Acquisition complete!")
    # emit signal to update button states
    signal_bus.acquisition_stopped.emit()
    # Acquisition completion signal is now emitted by the acquisition classes themselves

def handle_stop_acquisition(app_state, signal_bus):
    # Stop acquisition if running
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

        signal_bus.console_message.emit("Acquisition stopped")
        
        # emit signal to update button states
        signal_bus.acquisition_stopped.emit()
    
    # Stop local RPOC treatment if running
    if hasattr(signal_bus, 'local_rpoc_thread') and hasattr(signal_bus, 'local_rpoc_worker'):
        thread = signal_bus.local_rpoc_thread
        worker = signal_bus.local_rpoc_worker

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

        del signal_bus.local_rpoc_thread
        del signal_bus.local_rpoc_worker
        signal_bus.console_message.emit("Local RPOC treatment stopped")
    
    return 0

def handle_console_message(message, app_state, main_window):
    main_window.top_bar.add_console_message(message)
    return 0

def handle_modality_changed(text, app_state, main_window):
    # Use modality registry to get the correct modality key
    from pyrpoc.modalities import modality_registry
    
    modality = modality_registry.get_modality_by_name(text)
    if modality is not None:
        app_state.modality = modality.key
    else:
        # Fallback to lowercase text if modality not found
        app_state.modality = text.lower()

    handle_stop_acquisition(app_state,main_window.signals)

    app_state.current_data = None

    main_window.build_gui()
    
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
    # Refresh channel labels in multichannel display if it exists
    if hasattr(main_window, 'mid_layout') and hasattr(main_window.mid_layout, 'image_display_widget'):
        display_widget = main_window.mid_layout.image_display_widget
        if hasattr(display_widget, 'refresh_channel_labels'):
            display_widget.update_channel_names()
            display_widget.refresh_channel_labels()
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

def handle_mask_created(mask, app_state, main_window, signal_bus):
    # the mask is now handled by the individual channel widgets
    # this function is kept for backward compatibility but may not be used
    signal_bus.console_message.emit("RPOC mask created.")
    return 0

def handle_rpoc_channel_removed(channel_id, app_state, signal_bus):
    # remove the mask from app_state if it exists
    if hasattr(app_state, 'rpoc_mask_channels') and channel_id in app_state.rpoc_mask_channels:
        del app_state.rpoc_mask_channels[channel_id]
    
    # remove the DAQ channel info from app_state if it exists
    if hasattr(app_state, 'rpoc_script_channels') and channel_id in app_state.rpoc_script_channels:
        del app_state.rpoc_script_channels[channel_id]
    
    signal_bus.console_message.emit(f"RPOC channel {channel_id} removed.")
    return 0

def handle_local_rpoc_started(parameters, app_state, signal_bus):
    return 0
    # """Handle local RPOC treatment start - similar structure to handle_single_acquisition"""
    
    # # check if local RPOC treatment is already running
    # if hasattr(signal_bus, 'local_rpoc_thread') and hasattr(signal_bus, 'local_rpoc_worker'):
    #     signal_bus.console_message.emit("Local RPOC treatment already in progress. Please stop the current treatment first.")
    #     return 0
    
    # # Extract mask data and channel info from parameters
    # mask_data = parameters.pop('mask_data', None)
    # channel_id = parameters.pop('channel_id', None)
    
    # signal_bus.console_message.emit(f"Debug: Received parameters keys: {list(parameters.keys())}")
    # signal_bus.console_message.emit(f"Debug: Mask data received: {mask_data is not None}")
    # signal_bus.console_message.emit(f"Debug: Channel ID: {channel_id}")
    
    # if mask_data is None:
    #     signal_bus.console_message.emit("Error: No mask data available for local RPOC treatment")
    #     return 0
    
    # # Validate that we have a galvo instrument (required for treatment)
    # galvo_instruments = app_state.get_instruments_by_type('galvo')
    # if not galvo_instruments:
    #     signal_bus.console_message.emit("Error: Local RPOC treatment requires at least one galvo instrument. Please add a galvo scanner.")
    #     return 0
    
    # galvo = galvo_instruments[0]  # Use the first galvo instrument
    
    # signal_bus.console_message.emit(f"Local RPOC treatment started with {parameters['repetitions']} repetitions")
    # signal_bus.console_message.emit(f"Parameters: dwell_time={parameters['dwell_time']}μs, "
    #                               f"amplitude_x={parameters['amplitude_x']}V, "
    #                               f"amplitude_y={parameters['amplitude_y']}V, "
    #                               f"offset_x={parameters['offset_x']}V, "
    #                               f"offset_y={parameters['offset_y']}V, "
    #                               f"drift_x={parameters['offset_drift_x']}V, "
    #                               f"drift_y={parameters['offset_drift_y']}V")
    # signal_bus.console_message.emit(f"TTL Channel: {parameters.get('ttl_device', 'Dev1')}/{parameters.get('ttl_port_line', 'port0/line0')}")
    
    # # Log PFI line information if specified
    # pfi_line = parameters.get('pfi_line', 'None')
    # if pfi_line and pfi_line != 'None':
    #     signal_bus.console_message.emit(f"Timing: Using PFI line {pfi_line} for DO task timing")
    # else:
    #     signal_bus.console_message.emit("Timing: Using internal wiring (AO sample clock) for DO task timing")
    
    # # Create local RPOC object (similar to acquisition creation)
    # local_rpoc = None
    # try:
    #     local_rpoc = LocalRPOC(
    #         galvo=galvo,
    #         mask_data=mask_data,
    #         treatment_parameters=parameters,
    #         signal_bus=signal_bus
    #     )
    # except Exception as e:
    #     signal_bus.console_message.emit(f'Error creating local RPOC object: {e}')
    #     return 0
    
    # if local_rpoc is not None:
    #     # Show progress dialog
    #     from pyrpoc.rpoc.local_treatment import LocalRPOCProgressDialog
    #     progress_dialog = LocalRPOCProgressDialog(parameters['repetitions'])
    #     signal_bus.local_rpoc_progress_dialog = progress_dialog
        
    #     # Connect the cancel signal to stop the treatment
    #     progress_dialog.cancel_requested.connect(lambda: handle_local_rpoc_cancel(signal_bus))
        
    #     progress_dialog.show()
        
    #     # Use the same threading model as acquisition
    #     worker = AcquisitionWorker(local_rpoc, continuous=False)
    #     thread = QThread()
    #     worker.moveToThread(thread)
        
    #     local_rpoc.worker = worker
    #     worker.acquisition_finished.connect(lambda data: handle_local_rpoc_thread_finished(data, signal_bus, thread, worker))
    #     thread.started.connect(worker.run)
    #     thread.start()
        
    #     # Store references for cleanup (same as acquisition)
    #     signal_bus.local_rpoc_thread = thread
    #     signal_bus.local_rpoc_worker = worker
    # else:
    #     signal_bus.console_message.emit("Error: Failed to create local RPOC object")
    
    # return 0


def handle_local_rpoc_thread_finished(data, signal_bus, thread, worker):
    """Handle local RPOC treatment completion - similar to handle_acquisition_thread_finished"""
    # Clean up thread and worker (same pattern as acquisition)
    worker.stop()
    thread.quit()
    thread.wait()
    worker.deleteLater()
    thread.deleteLater()
    
    # Close progress dialog
    if hasattr(signal_bus, 'local_rpoc_progress_dialog'):
        signal_bus.local_rpoc_progress_dialog.set_completed()
        signal_bus.local_rpoc_progress_dialog.close()
        del signal_bus.local_rpoc_progress_dialog
    
    # Remove references
    if hasattr(signal_bus, 'local_rpoc_thread'):
        del signal_bus.local_rpoc_thread
    if hasattr(signal_bus, 'local_rpoc_worker'):
        del signal_bus.local_rpoc_worker
    
    signal_bus.console_message.emit("Local RPOC treatment completed!")
    return 0


def handle_local_rpoc_progress(repetition, app_state, signal_bus):
    """Handle local RPOC progress updates"""
    if hasattr(signal_bus, 'local_rpoc_progress_dialog'):
        signal_bus.local_rpoc_progress_dialog.update_progress(repetition)
    return 0

def handle_local_rpoc_cancel(signal_bus):
    """Handle local RPOC treatment cancellation"""
    # Stop the treatment if it's running
    if hasattr(signal_bus, 'local_rpoc_thread') and hasattr(signal_bus, 'local_rpoc_worker'):
        thread = signal_bus.local_rpoc_thread
        worker = signal_bus.local_rpoc_worker

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

        del signal_bus.local_rpoc_thread
        del signal_bus.local_rpoc_worker
        signal_bus.console_message.emit("Local RPOC treatment cancelled")
    
    # Close progress dialog
    if hasattr(signal_bus, 'local_rpoc_progress_dialog'):
        signal_bus.local_rpoc_progress_dialog.set_stopped()
        signal_bus.local_rpoc_progress_dialog.close()
        del signal_bus.local_rpoc_progress_dialog
    
    return 0


def handle_display_setup(display_class_name, app_state, main_window):
    try:
        if app_state.selected_display != display_class_name:
            app_state.selected_display = display_class_name
            main_window.rebuild_display()
            
            if hasattr(main_window, 'left_widget') and hasattr(main_window.left_widget, 'display_controls'):
                main_window.left_widget.display_controls.update_display_selection(display_class_name)
            
            if hasattr(main_window, 'signals'):
                main_window.signals.console_message.emit(f"Display updated to {display_class_name}")
        else:
            if hasattr(main_window, 'signals'):
                main_window.signals.console_message.emit("Display type unchanged, proceeding with acquisition")
        
        # Now prepare the display for acquisition using the modality's context
        modality = modality_registry.get_modality(app_state.modality)
        if modality:
            # Create acquisition context from the modality
            acquisition_context = modality.create_acquisition_context(app_state.acquisition_parameters)
            
            # Get the current display widget and prepare it
            if hasattr(main_window, 'mid_layout') and hasattr(main_window.mid_layout, 'image_display_widget'):
                display_widget = main_window.mid_layout.image_display_widget
                if hasattr(display_widget, 'handle_display_setup'):
                    display_widget.handle_display_setup()
                    main_window.signals.console_message.emit(f"Display prepared for {modality.name} acquisition")
                else:
                    main_window.signals.console_message.emit("Warning: Display widget doesn't support acquisition preparation")
        
    except Exception as e:
        if hasattr(main_window, 'signals'):
            main_window.signals.console_message.emit(f"Error setting up display: {e}")
        return 0
    
    return 1


def handle_data_received(data, app_state, main_window):
    """Handle individual data frame during acquisition"""
    try:
        # Route the data frame to the current display widget
        if hasattr(main_window, 'mid_layout') and hasattr(main_window.mid_layout, 'image_display_widget'):
            widget = main_window.mid_layout.image_display_widget
            if hasattr(widget, 'handle_data_received'):
                widget.handle_data_received(data)
            else:
                # Fallback to legacy method if new method not available
                if hasattr(main_window, 'signals'):
                    main_window.signals.console_message.emit("Warning: Display widget doesn't support new data frame method")
        
    except Exception as e:
        if hasattr(main_window, 'signals'):
            main_window.signals.console_message.emit(f"Error handling data frame: {e}")
        return 0
    
    return 1


def handle_acquisition_complete(app_state, main_window):
    """Handle acquisition completion"""
    try:
        # Signal that acquisition is complete
        if hasattr(main_window, 'signals'):
            main_window.signals.console_message.emit("Acquisition complete!")
            
            # Update button states
            main_window.signals.acquisition_stopped.emit()
        
    except Exception as e:
        if hasattr(main_window, 'signals'):
            main_window.signals.console_message.emit(f"Error handling acquisition completion: {e}")
        return 0
    
    return 1