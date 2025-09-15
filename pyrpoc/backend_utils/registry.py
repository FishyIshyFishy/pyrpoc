class Registry:
    def __init__(self, name: str, base_class: type):
        '''
        description:
            generic registry for which different component types can inhereit


            each registry enforces a decorator and a base class, so that using
            the decorator on each SubClass(BaseClass) immediately populates the registry
            on startup of the GUI

        args: 
            name: string name for the registry 
            base_class: the base_class that corresponds to the registry type

        attributes:
            entries: a dictionary of names and classes for each SubClass (e.g., 'camera': CameraInstrument)

        example:
            InstrumentRegistry(Registry):
                def __init__(self)
                    super().__init__(self, parent)
        '''
        self.name = name
        self.base_class = base_class # BaseInstrument, BaseModality, whatever
        self.entries = {}
        
    def register(self, key: str):
        '''
        description:
            register a class given a key corresponding to that class for its parent registry

            this function is used as a decorator. Upon import, the decorator will add the key
            to self.entries
        
        args:
            key: the name to store the class under (e.g., 'camera')

        returns:
            decorator: the decorator function to save the class into the registry

        example:
            @InstrumentRegistry.register('camera')
            class CameraInstrument(BaseInstrument):
                ...
        '''
        def decorator(cls):
            if not issubclass(cls, self.base_class):
                raise TypeError(f'{cls.__name__} must inheret from {self.base_class.__name__}')

            if key in self.entries:
                raise KeyError(f'{key!r} is already registered within the registry: {self.name}')
            
            self.entries[key] = cls
            return cls
        
        return decorator
    
    def get_registered(self):
        '''
        description:
            list all the registered things (for example, to directly list in GUI)

        args: none    
        
        returns: 
            list: a list of the keys in the entries dictionary

        example:
            InstrumentRegistry.get_registered()
            --> returns ['camera', 'stage', 'TCSPC']
        '''
        return list(self.entries.keys())
