from handy_core import IHandyNetwork

"""
This is the network that should be implemented
"""
class HandyConcrete(IHandyNetwork):
    pass


class HandyConcreteFake(IHandyNetwork):

    def predict_handwashing_time_live(self, frames:list) -> (bool, int):
        return False, 14

    def predict_handwashing_time(self, frames:list) -> int:
        return 14


# This module should have the object initialized
handy = HandyConcreteFake()