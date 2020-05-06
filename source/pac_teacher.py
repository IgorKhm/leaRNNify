import time

from dfa import DFA
from random_words import random_word_by_letter
from teacher import Teacher
from dfa_check import DFA_checker
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
            learner.new_counterexample(counter, True)

    def check_and_teach(self, learner, specification):
        learner.teacher = self
        checker = DFA_checker(learner.dfa, specification)
        while True:
            counter_from_spec = checker.check_for_counterexample()
            print(counter_from_spec)
            if counter_from_spec is None:
                counter_from_equiv = self.equivalence_query(learner.dfa)
                print(counter_from_equiv)
                if counter_from_equiv is None:
                    break
                else:
                    learner.new_counterexample(counter_from_equiv)
            else:
                if not self.model.is_word_in(counter_from_spec):
                    print('found counter mistake in the model: ',counter_from_spec)
                    break
                learner.new_counterexample(counter_from_spec)

            checker.model = learner.dfa

