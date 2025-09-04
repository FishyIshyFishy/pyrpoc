first create a new file `pyrpoc/modalities/your_modality.py`:

```python
from .base import BaseModality
from pyrpoc.displays.your_display import YourDisplayWidget
# from .your_modality_file import YourAcquisition  # acquisition now lives with modality
from typing import List, Type, Dict, Any

class YourModality(BaseModality):
    @property
    def name(self) -> str:
        return "Your Modality Name"
    
    @property
    def key(self) -> str:
        return "your_modality"
    
    @property
    def required_instruments(self) -> List[str]:
        return ["instrument_type1", "instrument_type2"]
    
    @property
    def compatible_displays(self) -> List[Type[BaseImageDisplayWidget]]:
        return [YourDisplayWidget]
    
    @property
    def required_parameters(self) -> Dict[str, Dict[str, Any]]:
        return {
            'param_name': {
                'type': 'int',  # 'int', 'float', 'bool', 'str', or 'choice'
                'default': 100,
                'range': (1, 1000),  # for numeric types
                'unit': 'units',  # optional
                'description': 'Human readable description'
            },
            'choice_param': {
                'type': 'choice',
                'default': 'option1',
                'choices': ['option1', 'option2', 'option3'],
                'description': 'Choice parameter'
            }
        }
    
    @property
    def acquisition_class(self) -> Type:
        return YourAcquisition
```

then add the modality to the registry in `pyrpoc/modalities/__init__.py`:

```python
from .your_modality import YourModality

modality_registry.register(YourModality())
```