import time

from dfa import DFA
from random_words import random_word_by_letter, random_word
from teacher import Teacher
from dfa_check import DFAChecker
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
            word = random_word(self.model.alphabet)
            if dfa.is_word_in(word) != self.model.is_word_in(word):
                return word
            # for i in range(int(number_of_rounds % 100)):
            #     words = [random_word(self.model.alphabet) for _ in range(100)]
            #     rnn_labels = self.model.is_words_in_batch(words) > 0.5
            #     dfa_labels = [dfa.is_word_in(word) for word in words]
            #     # word = random_word(self.model.alphabet)
            #     for rnn_label, dfa_label in zip(rnn_labels, dfa_labels):
            #         if rnn_label != dfa_label:
            #             return words[dfa_labels.index(dfa_label)]

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

    def check_and_teach(self, learner, checkers, timeout=900):
        learner.teacher = self
        start_time = time.time()

        while True:
            if time.time() - start_time > timeout:
                return
            print(time.time() - start_time)
            counters_from_specs = [(checker.check_for_counterexample(learner.dfa), checker.is_super_set) for checker in
                                   checkers]
            counter_from_spec = [None, None]
            for word in counters_from_specs:
                if word[0] is not None:
                    counter_from_spec = word
            if counter_from_spec[0] is None:
                counter_from_equiv = self.equivalence_query(learner.dfa)
                if counter_from_equiv is None:
                    return None
                else:
                    learner.new_counterexample(counter_from_equiv, set=True)

            else:
                if not (counter_from_spec[1] ^ (self.model.is_word_in(counter_from_spec[0]))):
                    print('found counter mistake in the model: ', counter_from_spec)
                    return counter_from_spec
                learner.new_counterexample(counter_from_spec[0], set=True)
