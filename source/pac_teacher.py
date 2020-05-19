import time
from collections import namedtuple

import matplotlib.pyplot as plt
import numpy as np

from dfa import DFA
from dfa_check import DFAChecker
from modelPadding import RNNLanguageClasifier
from random_words import random_word, confidence_interval_many, confidence_interval_many_for_reuse
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

        self.prev_examples = {}

        self.is_counter_example_in_batches = isinstance(self.model, RNNLanguageClasifier)

    def equivalence_query(self, dfa: DFA):
        """
        Tests whether the dfa is equivalent to the model by testing random words.
        If not equivalent returns an example
        """
        self._num_equivalence_asked = self._num_equivalence_asked + 1

        if dfa.is_word_in("") != self.model.is_word_in(""):
            return ""
        batch = []
        batch_size = 200
        number_of_rounds = int((self._log_delta - self._num_equivalence_asked) / self._log_one_minus_epsilon)
        # print(number_of_rounds)
        for i in range(number_of_rounds):
            word = random_word(self.model.alphabet)
            if self.model.is_word_in(word) != dfa.is_word_in(word):
                return word
            # batch.append(word)
            # i += 1
            # if i % batch_size == 0:
            #     for x, y, w in zip(self.model.is_words_in_batch(batch) > 0.5, [dfa.is_word_in(w) for w in batch],
            #                        batch):
            #         if x != y:
            #             return w
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
                print(time.time() - start_time)
                return
            print(i)
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

    def teach_and_trace(self, student, dfa_model, timeout=900):
        output, smaples, answers = confidence_interval_many_for_reuse([dfa_model, self.model, student.dfa], random_word,
                                                                      width=0.1, confidence=0.1)
        dist_to_dfa_vs = []
        dist_to_rnn_vs = []
        num_of_states = []

        # points.append(DataPoint(len(student.dfa.states), output[0, 2], output[1, 2]))

        a = None
        student.teacher = self
        i = 0
        start_time = time.time()
        t100 = start_time
        while True:
            if time.time() - start_time > timeout:
                break
            i = i + 1
            if i % 100 == 0:
                print("this is the {}th round".format(i))
                print("{} time has passed from the begging and {} from the last 100".format(time.time() - start_time,
                                                                                            time.time() - t100))
                t100 = time.time()
            counter = self.equivalence_query(student.dfa)
            if counter is None:
                break
            student.new_counterexample(counter, do_hypothesis_in_batches=False)

            print('compute dist')
            output, _, answers = confidence_interval_many_for_reuse([dfa_model, self.model, student.dfa], random_word,
                                                                    answers, samples=smaples, width=0.1, confidence=0.1)
            # points.append(DataPoint(len(student.dfa.states), output[0, 2], output[1, 2]))

            dist_to_dfa_vs.append(output[0][2])
            dist_to_rnn_vs.append(output[1][2])
            num_of_states.append(len(student.dfa.states))
            print('done compute dist')

        # plt.plot(num_of_states, dist_to_dfa_vs, label="DvD",color='green', linestyle='dashed')
        # plt.title('original dfa vs extracted dfa')
        #
        # plt.plot(num_of_states, dist_to_rnn_vs, label="RvD",)
        # plt.title('rnn vs extracted dfa')
        # plt.legend()
        # plt.figure()

        #
        fig = plt.figure(dpi=1200)
        ax = fig.add_subplot(2, 1, 1)

        ax.plot(num_of_states, dist_to_dfa_vs, color='blue', lw=2)

        ax.set_yscale('log')

        plt.show()

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
