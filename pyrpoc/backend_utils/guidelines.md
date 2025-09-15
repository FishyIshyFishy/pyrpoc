# Project Coding Guidelines

These are just helpful guidelines for myself as I try to be super duper organized this go-around. I guess anyone else who sees this could also benefit if they want to add to the project. 

---

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