from abc import ABC, abstractmethod

class IHandyNetwork(ABC):
    """
    Interface for a network that is able to predict handwashing times
    """

    @abstractmethod
    def predict_handwashing_time_live(self, frames:list) -> bool, int:
        pass

    @abstractmethod
    def predict_handwashing_time(self, frames:list) -> int:
        pass