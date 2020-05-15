import time
from collections import namedtuple

import numpy as np

from dfa import DFA
from dfa_check import DFAChecker
from modelPadding import LSTMLanguageClasifier
from random_words import random_word
from teacher import Teacher


class PACTeacher(Teacher):

    def __init__(self, model: DFA, epsilon=0.001, delta=0.001):
        assert ((epsilon <= 1) & (delta <= 1))
        Teacher.__init__(self, model)
        self.epsilon = epsilon
        self.delta = delta
        self._log_delta = np.log(delta)
        self._log_one_minus_epsilon = np.log(1 - epsilon)
        self._num_equivalence_asked = 0

        self.is_counter_example_in_batches = isinstance(self.model, LSTMLanguageClasifier)

    def equivalence_query(self, dfa: DFA):
        """
        Tests whether the dfa is equivalent to the model by testing random words.
        If not equivalent returns an example
        """
        self._num_equivalence_asked = self._num_equivalence_asked + 1

        if dfa.is_word_in("") != self.model.is_word_in(""):
            return ""

        number_of_rounds = int((self._log_delta - self._num_equivalence_asked) / self._log_one_minus_epsilon)
        for i in range(number_of_rounds):
            word = random_word(self.model.alphabet)
            if dfa.is_word_in(word) != self.model.is_word_in(word):
                return word
            # ------------------------------------------------------
            # Code for going through batches and not word by word:#
            #
            # for i in range(int(number_of_rounds % 100)):
            #     words = [random_word(self.model.alphabet) for _ in range(100)]
            #     rnn_labels = self.model.is_words_in_batch(words) > 0.5
            #     dfa_labels = [dfa.is_word_in(word) for word in words]
            #     # word = random_word(self.model.alphabet)
            #     for rnn_label, dfa_label in zip(rnn_labels, dfa_labels):
            #         if rnn_label != dfa_label:
            #             return words[dfa_labels.index(dfa_label)]
            # ------------------------------------------------------
        return None

    def membership_query(self, word):
        return self.model.is_word_in(word)

    def teach(self, learner, timeout=900):
        learner.teacher = self
        i = 0
        start_time = time.time()
        t100 = start_time
        while True:
            if time.time() - start_time > timeout:
                return
            i = i + 1
            if i % 100 == 0:
                print("this is the {}th round".format(i))
                print("{} time has passed from the begging and {} from the last 100".format(time.time() - start_time,
                                                                                            time.time() - t100))
                t100 = time.time()

            counter = self.equivalence_query(learner.dfa)
            if counter is None:
                break
            learner.new_counterexample(counter, do_hypothesis_in_batches=False)

    def check_and_teach(self, learner, checkers: [DFAChecker], timeout=900):
        learner.teacher = self
        start_time = time.time()
        Counter_example = namedtuple('Counter_example', ['word', 'is_super'])

        while True:
            if time.time() - start_time > timeout:
                return
            print(time.time() - start_time)

            counter_example = Counter_example(None, None)

            # Searching for counter examples in the spec:
            counters_examples = (Counter_example(checker.check_for_counterexample(learner.dfa), checker.is_super_set)
                                 for checker in checkers)
            for example in counters_examples:
                if example.word is not None:
                    counter_example = example
                    break
            if counter_example.word is not None:
                if counter_example.is_super != (self.model.is_word_in(counter_example.word)):
                    learner.new_counterexample(counter_example[0], do_hypothesis_in_batches=False)
                else:
                    print('found counter mistake in the model: ', counter_example)
                    return counter_example

            # Searching for counter examples in the the model:
            else:

                counter_example = self.equivalence_query(learner.dfa)
                if counter_example is None:
                    return None
                else:
                    learner.new_counterexample(counter_example, do_hypothesis_in_batches=False)
