import cProfile
import collections
import os
import time

import numpy as np

from benchmarking import rand_benchmark, run_rand_benchmarks
from dfa import DFA, load_dfa_dot, random_dfa, intersection
from dfa import save_dfa_as_part_of_model
from dfa_check import DFAChecker
from exact_teacher import ExactTeacher
from learner_decison_tree import DecisionTreeLearner
from modelPadding import LSTMLanguageClasifier
from pac_teacher import PACTeacher
from random_words import random_word, confidence_interval_many


def create_file_for_dfa():
    dfa = DFA("s0r1", {"s0r1"}, {"s0r1": {"msg0": "s0r0", "msg1": "sink", "ack0": "sink", "ack1": "s0r1"},
                                 "s0r0": {"msg0": "s0r0", "msg1": "sink", "ack0": "s1r0", "ack1": "sink"},
                                 "s1r0": {"msg0": "sink", "msg1": "s1r1", "ack0": "s1r0", "ack1": "sink"},
                                 "s1r1": {"msg0": "sink", "msg1": "s1r1", "ack0": "sink", "ack1": "s0r1"},
                                 "sink": {"msg0": "sink", "msg1": "sink", "ack0": "sink", "ack1": "sink"}})
    return dfa


print("Begin")

dfa = create_file_for_dfa()
dfa.draw_nicely(name="to learn")
spec = DFA("1", {"1"}, {"1": {"msg0": "2", "msg1": "sink", "ack0": "sink", "ack1": "1"},
                        "2": {"msg0": "2", "msg1": "2", "ack0": "2", "ack1": "1"},
                        "sink": {"msg0": "sink", "msg1": "sink", "ack0": "sink", "ack1": "sink"}})

spec2 = DFA("1", {"1", "5"}, {"1": {"msg0": "2", "msg1": "2", "ack0": "2", "ack1": "1"},
                              "2": {"msg0": "3", "msg1": "3", "ack0": "3", "ack1": "3"},
                              "3": {"msg0": "4", "msg1": "4", "ack0": "4", "ack1": "4"},
                              "4": {"msg0": "5", "msg1": "5", "ack0": "5", "ack1": "5"},
                              "5": {"msg0": "5", "msg1": "5", "ack0": "5", "ack1": "5"}})

spec3 = DFA("0", {"1"}, {"0": {"msg0": "2", "msg1": "5", "ack0": "5", "ack1": "5"},
                         "1": {"msg0": "2", "msg1": "5", "ack0": "5", "ack1": "5"},
                         "2": {"msg0": "5", "msg1": "5", "ack0": "3", "ack1": "5"},
                         "3": {"msg0": "5", "msg1": "4", "ack0": "5", "ack1": "5"},
                         "4": {"msg0": "5", "msg1": "5", "ack0": "5", "ack1": "1"},
                         "5": {"msg0": "5", "msg1": "5", "ack0": "5", "ack1": "5"}})
# rand_benchmark("../models/rand/test2/")

run_rand_benchmarks()
# spec = intersection(spec, spec2)
# teacher_pac1 = ExactTeacher(spec)
# student1 = DecisionTreeLearner(teacher_pac1)
# teacher_pac1.teach(student1)
# spec = student1.dfa
#
# specClass = DFA_checker(dfa, spec, is_super_set=True)
# specClass2 = DFA_checker(dfa, spec3, is_super_set=False)
#
# learn_and_check(dfa, [specClass, specClass2])
