import time

from dfa import DFA
from random_words import random_word_by_letter
from teacher import Teacher
import numpy as np


class PACTeacher(Teacher):

    def __init__(self, model: DFA, epsilon=0.001, delta=0.001):
        assert ((epsilon <= 1) & (delta <= 1))
        Teacher.__init__(self, model)
        self.epsilon = epsilon
        self.delta = delta
        self._log_delta = np.log(delta)
        self._log_one_minus_epsilon = np.log(1 - epsilon)
        self._num_equivalence_asked = 0

    def equivalence_query(self, dfa):
        self._num_equivalence_asked = self._num_equivalence_asked + 1

        if dfa.is_word_in("") != self.model.is_word_in(""):
            return ""

        number_of_rounds = int((self._log_delta - self._num_equivalence_asked) / self._log_one_minus_epsilon)
        for i in range(number_of_rounds):
            dfa.reset_current_to_init()
            # self.model.reset_current_to_init()
            word = ""
            for letter in random_word_by_letter(self.model.alphabet):
                word = word + letter
                # if dfa.is_word_letter_by_letter(letter) != self.model.is_word_in(word):
                #     return word
            if dfa.is_word_in(word) != self.model.is_word_in(word):
                # print("in DFA: " + str(dfa.is_word_in(word)))
                # print("counter example: " + word)
                return word
        return None

    def membership_query(self, w):
        return self.model.is_word_in(w)

    def teach(self, learner, timeout=900):
        learner.teacher = self
        i = 0
        t = time.time()
        t100 = t
        while True:
            if time.time() - t > timeout:
                return
            i = i + 1
            if i % 100 == 0:
                print("this is the {}th round".format(i))
                print("{} time has passed from the beging and {} from the last 100".format(time.time() - t,
                                                                                           time.time() - t100))
                t100 = time.time()
            counter = self.equivalence_query(learner.dfa)
            if counter is None:
                break
            learner.new_counterexample(counter, False)

    # n > (log(delta) -num_round) / log(1-epsilon)
