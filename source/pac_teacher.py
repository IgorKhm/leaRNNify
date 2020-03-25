from dfa import DFA
from random_words import random_word_by_letter
from teacher import Teacher


class PACTeacher(Teacher):

    def __init__(self, model: DFA):
        Teacher.__init__(self, model)
        self._num_equivalence_asked = 0

    def equivalence_query(self, dfa):
        self._num_equivalence_asked = self._num_equivalence_asked + 1
        if dfa.is_word_in("") != self.model.is_word_in(""):
            return ""
        print(self._num_equivalence_asked)
        for i in range(4 * self._num_equivalence_asked):
            dfa.reset_current_to_init()
            self.model.reset_current_to_init()
            word = ""
            for letter in random_word_by_letter(self.model.alphabet, 1 / (10 * self._num_equivalence_asked)):
                word = word + letter
                if dfa.is_word_letter_by_letter(letter) != self.model.is_word_letter_by_letter(letter):
                    return word
        return None

    def membership_query(self, w):
        return self.model.is_word_in(w)

    def teach(self, learner):
        learner.teacher = self
        while True:
            counter = self.equivalence_query(learner.dfa)
            if counter is None:
                break
            learner.new_counterexample(counter)
