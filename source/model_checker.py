from abc import ABC, abstractmethod
from dfa import DFA


# Currently can only check DFAs
class ModelChecker:

    def __init__(self, model: DFA, specification):
        self.model = model
        self.specification = specification

    # check is the model satisfies the specification, else returns a counterexample
    # Can be divided into two functions check and counterexample, but now not necessary
    @abstractmethod
    def check_counterexamples():
        raise NotImplementedError()
