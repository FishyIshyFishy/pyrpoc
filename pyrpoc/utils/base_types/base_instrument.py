from abc import ABC, abstractmethod

class BaseInstrument(ABC):
    def __init__(self, name: str):
        self.name = name