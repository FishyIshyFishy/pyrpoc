# important notes to actually remember
## signal structuring
there are 5 sets of signals. might be good to add display_signals as well, since it's kind of split up by the code folder each signal interacts with
  - ui_signals, which has a reference to the other 4, and is the controller
  - app_state signals, to maintain the state for config loading/saving/error logging
  - acq_signals, which handles interaction with modalities/
  - instr_signals, which handles interaction with instruments/
  - laser_mod_signals, which handles interaction with laser_modulations/
  
the goal of this structure is to force the GUI to only interact with ui_signals, and then have ui_signals interact with the backend

## base modality
 so far, the pipeline for acquisition after acq_signals hands the base modality a context is
 1. validate parameters, instruments, and laser mods
 2. setup display if not already setup (i.e., display to use is different from current)
 3. initialize thread and pass the pure objects into perform_acquisition() within the thread
 4. have the thread send to display
 5. send final data object or something to say it's wrapped up

as i write this i'm realizing this is another argument for display signals, because the BaseModality can refer to acq_signals which ties directly to display signals.
  

# notes from a while ago to enforce coding guidelines for myself
## Docstrings

Use docstrings (`'''blah blah blah'''`) for all functions, classes, and important methods.  

```python
def function(self, arg1: int, arg2: int) -> int:
    """
    description:
        what this function does

    args:
        arg1: blah blah blah
        arg2: blah blah blah

    returns:
        something: blah blah blah

    example:
        something = function(3, 6)
    """
    return arg1 + arg2
````
## Signals

* Each group of signals has their own `QObject` subclass
* Each group should live in its own file under `signals/`
* Only pass the relevant signal group into each service or widget
* Never create one giant “signal bus.”


## Registries

Registries keep the design modular. Adding a new instrument, display, modality, or laser control should not require editing the GUI or central code.

* One registry per category:

  * `InstrumentRegistry`
  * `DisplayRegistry`
  * `ModalityRegistry`
  * `LaserModulationRegistry`

* Each registry enforces a base class (`BaseInstrument`, `BaseDisplay`, etc.).

* Classes self-register using a decorator:

  ```python
  @InstrumentRegistry.register('camera')
  class CameraInstrument(BaseInstrument):
      ...
  ```

* Make sure each folder's `__init__.py` is actually populated, so imports actually happen, so that decorators actually get called


## Project Layout

```
myapp/
├── backend_utils/    # organizational stuff for codebase structuring
├── demos/            # playing around with acquisition side stuff
├── displays/    
├── gui/              # the actual GUI code, split into different files for each part  
├── instruments/     
├── laser_modulations/     
├── modalities/        # this is the only place that knows different modalities exist     
├── signals/          # pyqtSignal classes grouped by general purpose 
└── main.py            # main entry point
```