import cProfile
import collections
import os
import time

import numpy as np

from benchmarking import rand_benchmark, run_rand_benchmarks, learn_multiple_times, run_multiple_spec_on_ltsm,learn_dfa
from dfa import DFA, load_dfa_dot, random_dfa, dfa_intersection
from dfa import save_dfa_as_part_of_model
from dfa_check import DFAChecker
from exact_teacher import ExactTeacher
from learner_decison_tree import DecisionTreeLearner
from modelPadding import LSTMLanguageClasifier
from pac_teacher import PACTeacher
from random_words import random_word, confidence_interval_many



def e_commerce_dfa():
    dfa = DFA("0", {"0,2,3,4,5"},
              {"0": {"os": "2", "gAP": "4", "gSC": "1", "bPSC": "1", "ds": "1", "eSC": "1", "aPSC": "1"},
               "1": {"os": "1", "gAP": "1", "gSC": "1", "bPSC": "1", "ds": "1", "eSC": "1", "aPSC": "1"},
               "2": {"os": "2", "gAP": "3", "gSC": "2", "bPSC": "1", "ds": "0", "eSC": "2", "aPSC": "1"},
               "3": {"os": "3", "gAP": "3", "gSC": "3", "bPSC": "1", "ds": "4", "eSC": "3", "aPSC": "5"},
               "4": {"os": "3", "gAP": "4", "gSC": "1", "bPSC": "1", "ds": "1", "eSC": "1", "aPSC": "1"},
               "5": {"os": "3", "gAP": "5", "gSC": "5", "bPSC": "3", "ds": "4", "eSC": "3", "aPSC": "5"}})
    return dfa


def alternating_bit_dfa():
    dfa = DFA("s0r1", {"s0r1"}, {"s0r1": {"msg0": "s0r0", "msg1": "sink", "ack0": "sink", "ack1": "s0r1"},
                                 "s0r0": {"msg0": "s0r0", "msg1": "sink", "ack0": "s1r0", "ack1": "sink"},
                                 "s1r0": {"msg0": "sink", "msg1": "s1r1", "ack0": "s1r0", "ack1": "sink"},
                                 "s1r1": {"msg0": "sink", "msg1": "s1r1", "ack0": "sink", "ack1": "s0r1"},
                                 "sink": {"msg0": "sink", "msg1": "sink", "ack0": "sink", "ack1": "sink"}})
    return dfa




print("Begin")

# dfa = create_file_for_dfa()
# dfa.draw_nicely(name="to learn")
# spec = DFA("1", {"1"}, {"1": {"msg0": "2", "msg1": "sink", "ack0": "sink", "ack1": "1"},
#                         "2": {"msg0": "2", "msg1": "2", "ack0": "2", "ack1": "1"},
#                         "sink": {"msg0": "sink", "msg1": "sink", "ack0": "sink", "ack1": "sink"}})
#
# spec2 = DFA("1", {"1", "5"}, {"1": {"msg0": "2", "msg1": "2", "ack0": "2", "ack1": "1"},
#                               "2": {"msg0": "3", "msg1": "3", "ack0": "3", "ack1": "3"},
#                               "3": {"msg0": "4", "msg1": "4", "ack0": "4", "ack1": "4"},
#                               "4": {"msg0": "5", "msg1": "5", "ack0": "5", "ack1": "5"},
#                               "5": {"msg0": "5", "msg1": "5", "ack0": "5", "ack1": "5"}})
#
# spec3 = DFA("0", {"1"}, {"0": {"msg0": "2", "msg1": "5", "ack0": "5", "ack1": "5"},
#                          "1": {"msg0": "2", "msg1": "5", "ack0": "5", "ack1": "5"},
#                          "2": {"msg0": "5", "msg1": "5", "ack0": "3", "ack1": "5"},
#                          "3": {"msg0": "5", "msg1": "4", "ack0": "5", "ack1": "5"},
#                          "4": {"msg0": "5", "msg1": "5", "ack0": "5", "ack1": "1"},
#                          "5": {"msg0": "5", "msg1": "5", "ack0": "5", "ack1": "5"}})
# rand_benchmark("../models/rand/test2/")

#
# learn_multiple_times(alternating_bit_dfa(), "../models/alternating_bit/lstm")
# learn_multiple_times(alternating_bit_dfa(), "../models/e_commerce/lstm")


############################################
# Alternating_bit_dfa tests:
############################################
alternating_bit_tests = []
messages_alternating = []
messages_alternating.append("1) after a sequence of msg0 there has to come ack0:")
spec = DFA("1", {"1"}, {"1": {"msg0": "2", "msg1": "1", "ack0": "1", "ack1": "1"},
                        "2": {"msg0": "2", "msg1": "sink", "ack0": "1", "ack1": "sink"},
                        "sink": {"msg0": "sink", "msg1": "sink", "ack0": "sink", "ack1": "sink"}})
alternating_bit_tests.append(spec)

messages_alternating.append("2) after a sequence of msg1 there has to come ack1:")
spec = DFA("1", {"1"}, {"1": {"msg0": "1", "msg1": "2", "ack0": "1", "ack1": "1"},
                        "2": {"msg0": "sink", "msg1": "2", "ack0": "sink", "ack1": "1"},
                        "sink": {"msg0": "sink", "msg1": "sink", "ack0": "sink", "ack1": "sink"}})
alternating_bit_tests.append(spec)

messages_alternating.append("3) after a sequence of ack0 there has to come msg1:")
spec = DFA("1", {"1"}, {"1": {"msg0": "1", "msg1": "1", "ack0": "2", "ack1": "1"},
                        "2": {"msg0": "sink", "msg1": "1", "ack0": "2", "ack1": "sink"},
                        "sink": {"msg0": "sink", "msg1": "sink", "ack0": "sink", "ack1": "sink"}})
alternating_bit_tests.append(spec)

messages_alternating.append("4) If there was msg0 there has to be msg1:")
spec = DFA("1", {"1"}, {"1": {"msg0": "2", "msg1": "1", "ack0": "1", "ack1": "1"},
                        "2": {"msg0": "2", "msg1": "1", "ack0": "2", "ack1": "2"}})
alternating_bit_tests.append(spec)

messages_alternating.append("5) If there was ack0 there has to be ack1:")
spec = DFA("1", {"1"}, {"1": {"msg0": "1", "msg1": "1", "ack0": "2", "ack1": "1"},
                        "2": {"msg0": "2", "msg1": "2", "ack0": "2", "ack1": "1"}})
alternating_bit_tests.append(spec)
############################################
############################################


############################################
# e_commerce tests:
############################################
ec_commerce_tests = []
# 1) you have to have an open session in order to do anything except opening a session or geting available product:
spec = DFA("1", {"1,2"}, {"1": {"os": "2", "gAP": "1", "gSC": "0", "bPSC": "0", "ds": "0", "eSC": "0", "aPSC": "0"},
                          "2": {"os": "2", "gAP": "2", "gSC": "2", "bPSC": "2", "ds": "1", "eSC": "2", "aPSC": "2"},
                          "0": {"os": "0", "gAP": "0", "gSC": "0", "bPSC": "0", "ds": "0", "eSC": "0", "aPSC": "0"}})
ec_commerce_tests.append(spec)
# 2) you have to get available products before adding them to your cart or buying them:
spec = DFA("1", {"1,2"}, {"1": {"os": "1", "gAP": "2", "gSC": "1", "bPSC": "0", "ds": "1", "eSC": "1", "aPSC": "0"},
                          "2": {"os": "2", "gAP": "2", "gSC": "2", "bPSC": "2", "ds": "2", "eSC": "2", "aPSC": "2"},
                          "0": {"os": "0", "gAP": "0", "gSC": "0", "bPSC": "0", "ds": "0", "eSC": "0", "aPSC": "0"}})
ec_commerce_tests.append(spec)
# 3) you need to have products in your shopping cart before buying them buying them:
spec = DFA("1", {"1,2"}, {"1": {"os": "1", "gAP": "1", "gSC": "1", "bPSC": "0", "ds": "1", "eSC": "1", "aPSC": "1"},
                          "2": {"os": "2", "gAP": "2", "gSC": "2", "bPSC": "1", "ds": "2", "eSC": "1", "aPSC": "2"},
                          "0": {"os": "0", "gAP": "0", "gSC": "0", "bPSC": "0", "ds": "0", "eSC": "0", "aPSC": "0"}})
ec_commerce_tests.append(spec)
############################################
############################################


print("Begin")

# dfa = create_file_for_dfa()
# dfa.draw_nicely(name="to learn")
# spec = DFA("1", {"1"}, {"1": {"msg0": "2", "msg1": "sink", "ack0": "sink", "ack1": "1"},
#                         "2": {"msg0": "2", "msg1": "2", "ack0": "2", "ack1": "1"},
#                         "sink": {"msg0": "sink", "msg1": "sink", "ack0": "sink", "ack1": "sink"}})
#
# spec2 = DFA("1", {"1", "5"}, {"1": {"msg0": "2", "msg1": "2", "ack0": "2", "ack1": "1"},
#                               "2": {"msg0": "3", "msg1": "3", "ack0": "3", "ack1": "3"},
#                               "3": {"msg0": "4", "msg1": "4", "ack0": "4", "ack1": "4"},
#                               "4": {"msg0": "5", "msg1": "5", "ack0": "5", "ack1": "5"},
#                               "5": {"msg0": "5", "msg1": "5", "ack0": "5", "ack1": "5"}})
#
# spec3 = DFA("0", {"1"}, {"0": {"msg0": "2", "msg1": "5", "ack0": "5", "ack1": "5"},
#                          "1": {"msg0": "2", "msg1": "5", "ack0": "5", "ack1": "5"},
#                          "2": {"msg0": "5", "msg1": "5", "ack0": "3", "ack1": "5"},
#                          "3": {"msg0": "5", "msg1": "4", "ack0": "5", "ack1": "5"},
#                          "4": {"msg0": "5", "msg1": "5", "ack0": "5", "ack1": "1"},
#                          "5": {"msg0": "5", "msg1": "5", "ack0": "5", "ack1": "5"}})
# rand_benchmark("../models/rand/test2/")

#
# learn_multiple_times(alternating_bit_dfa(), "../models/alternating_bit/lstm")
# learn_multiple_times(alternating_bit_dfa(), "../models/e_commerce/lstm")
dfa = alternating_bit_dfa()
benchmarks = {}
ltsm = learn_dfa(dfa, benchmarks,
                 hidden_dim=50,
                 num_layers=2,
                 epoch = 10,
                 num_of_exm_per_length=20000,
                 word_training_length=len(dfa.states) + 10)

print("Runnning tests on alternating_bit layers = 2 hidden din - 20:")
# ltsm.load_rnn("../models/alternating_bit/lstm/l-2__h-20")
run_multiple_spec_on_ltsm(ltsm,  alternating_bit_tests,messages_alternating)
print("################################################################")


# print("Runnning tests on alternating_bit layers = 5 hidden din - 50:")
# ltsm.load_rnn("../models/alternating_bit/lstm/l-5__h-50")
# run_multiple_spec_on_ltsm(ltsm,  alternating_bit_tests,messages_alternating)
# print("################################################################")
#
#
# print("Runnning tests on alternating_bit layers = 10 hidden din - 100:")
# ltsm.load_rnn("../models/alternating_bit/lstm/l-10__h-100")
# run_multiple_spec_on_ltsm(ltsm,  alternating_bit_tests,messages_alternating)
# print("################################################################")

#
#
# run_rand_benchmarks()
# # spec = intersection(spec, spec2)
# # teacher_pac1 = ExactTeacher(spec)
# student1 = DecisionTreeLearner(teacher_pac1)
# teacher_pac1.teach(student1)
# spec = student1.dfa
#
# specClass = DFA_checker(dfa, spec, is_super_set=True)
# specClass2 = DFA_checker(dfa, spec3, is_super_set=False)
#
# learn_and_check(dfa, [specClass, specClass2])
