# data management for state storage on startup

Everything relevant to the storage of the appstate should be in `pyrpoc/domain/appstate.py`. There is a list of InstrumentStates, OptoControlStates, DisplayStates, as well as a single ModalityState and GuiLayoutState. Each instance state contains whatever information is relevant to that instance that needs to be held for the backend OR the frontend

The services then only read or write from those states. Creation of a new instrument/display/modality should add into the list of those state instances owned by the AppState. 

Session information is stored in a session.json, see session_repository.py. SessionCodec.from_json_dict() then parses the json into a usable AppState variable to load in. 

Session is autosaved via session_coordinator.py with a QTimer.
