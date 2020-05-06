from dfa import DFA
from model_checker import ModelChecker


class DFA_checker(ModelChecker):

    def __init__(self, model: DFA, specification: DFA):
        super().__init__(model, specification)
        if model.alphabet != specification.alphabet:
            raise Exception("Model and specification have different alphabets")
        else:
            self.alphabet = model.alphabet

    def check_for_counterexample(self):
        dfs_stack = []
        word_path = ""
        dfs_stack.append([(self.model.init_state, self.specification.init_state), word_path])
        visited = []

        while len(dfs_stack) != 0:
            [(model_state, spec_state), word_path] = dfs_stack.pop(0)
            if ((model_state, spec_state) in visited):
                continue
            else:
                visited.append((model_state, spec_state))

            if (self.model.is_final_state(model_state) and not (self.specification.is_final_state(spec_state))):
                return word_path

            for letter in self.alphabet:
                dfs_stack.append([(model.next_state_by_letter(model_state, letter), \
                                   specification.next_state_by_letter(spec_state, letter)), word_path + letter])

        return None
