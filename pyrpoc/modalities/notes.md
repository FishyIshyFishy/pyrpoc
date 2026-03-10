
# Registry System Notes

This document summarizes the design decisions and reasoning behind the
registry system for pyrpoc.

---

## Core Idea

- A **generic `Registry`** class handles:
  - Type enforcement (`issubclass` check against a base class).
  - Decorator-based auto-registration.
  - Key uniqueness (no duplicate entries).
  - Lookup of registered classes.

- Specialized registries (e.g. `ModalityRegistry`, `InstrumentRegistry`,
  `LaserModulationRegistry`) subclass `Registry` to add type-specific
  utilities and behavior.

---

## Why Specialized Registries?

Even though `Registry` is generic, creating a dedicated subclass for each
type of component (e.g. `ModalityRegistry`) gives us a centralized place
to add **domain-specific logic**.

### Common Extensions

1. **Type-Specific Constructors**
   ```python
   def create(self, key: str, **kwargs) -> BaseModality:
       return self.entries[key](**kwargs)


2. **GUI Integration Hooks**

   ```python
   def as_qmenu(self, parent=None):
       # Returns a QMenu with each registry entry as an action
   ```

   * Useful for managers / menus / dropdowns.

3. **Default / Preferred Logic**

   ```python
   def get_default(self) -> BaseModality:
       return self.create(next(iter(self.entries)))
   ```

4. **Validation and Constraints**

   ```python
   def validate_active_set(self, active: list[str]) -> bool:
       # Ensure only one modality is active at once
   ```

5. **Serialization / Persistence**

   ```python
   def to_config(self, instance: BaseModality) -> dict
   def from_config(self, config: dict) -> BaseModality
   ```

   * For saving/loading GUI sessions.

---

## Folder Structure Pattern

Each "thing" (modalities, instruments, laser\_modulation, etc.) should
have a similar package layout:

```
somethings/
│
├── base_something.py          # defines BaseSomething
├── something_registry.py      # defines SomethingRegistry + singleton
├── example_something.py       # concrete implementation
│
└── widgets/
    ├── base_something_widget.py   # defines BaseSomethingWidget
    └── example_widget.py          # per-implementation widget
```

* **`base_X.py`** → abstract interface all implementations must inherit from.
* **`X_registry.py`** → concrete registry subclass + singleton instance.
* **Implementations** → one file per subclass, decorated with `@X_registry.register(...)`.
* **`widgets/`** → base widget + per-subclass widgets for GUI integration.

---

## Example Flow

```python
# confocal.py
@modality_registry.register('confocal')
class ConfocalModality(BaseModality):
    def start(self):
        print("Confocal started")
```

```python
# usage
from modalities.modality_registry import modality_registry
import modalities.confocal  # ensures registration

print(modality_registry.get_registered())
# -> ['confocal']

confocal = modality_registry.create('confocal', name="Confocal #1")
confocal.start()
```

---

## Key Takeaway

* **`Registry`** = generic, reusable foundation.
* **`ModalityRegistry` / `InstrumentRegistry` / `LaserModulationRegistry`** = smart, type-aware managers.
* Specialized registries should grow to handle instantiation, GUI integration, validation, and persistence **specific to their domain**.
