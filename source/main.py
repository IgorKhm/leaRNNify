import os
import time

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import cProfile
import torch

from dfa import load_dfa_dot, random_dfa
from learner_decison_tree import DecisionTreeLearner
from modelPadding import LSTMLanguageClasifier, test_rnn
from pac_teacher import PACTeacher
from random_words import confidence_interval, random_word, confidence_interval_many

from dfa import save_dfa_dot, save_dfa_as_part_of_model
from exact_teacher import ExactTeacher
from lstar.Extraction import extract


def target(w):
    return w[0] == w[-1]


class lan():
    def is_word_in(self, w):
        return target(w)


def learn_dfa_and_compare_distance(dir):
    for folder in os.walk(dir):
        if folder[0] == dir:
            continue
        dfaOrigin = load_dfa_dot(folder[0] + r"\dfa.dot")
        ltsm = LSTMLanguageClasifier()
        ltsm.load_rnn(folder[0])

        lstar_dfa = extract(ltsm, time_limit=600, initial_split_depth=20)
        teacher_pac = PACTeacher(ltsm)
        student_pac = DecisionTreeLearner(teacher_pac)
        teacher_pac.teach(student_pac, 600)

        a, samples = confidence_interval_many([dfaOrigin, ltsm, student_pac], random_word,
                                              epsilon=0.005)
        print(a)

    return


def main_train_RNNS():
    for i in range(9, 10):
        alphabet = "abcd"
        dfa_rand = random_dfa(alphabet, 10, 30, 1, 10)
        print(dfa_rand)
        teacher_exact = ExactTeacher(dfa_rand)
        student_exact = DecisionTreeLearner(teacher_exact)
        teacher_exact.teach(student_exact)
        dfa_rand = student_exact.dfa
        r = lan()
        print(r.is_word_in("bababababb"))
        print(dfa_rand)
        dfa_rand.draw_nicely(name="_testing_stuff")
        save_dfa_as_part_of_model("models2/" + str(i), dfa_rand, True)
        starttime = time.time()
        model = LSTMLanguageClasifier()
        model.train_a_lstm(alphabet, target, hidden_dim=10, num_layers=1, embedding_dim=5,
                           num_of_exm_per_lenght=20000, batch_size=20, epoch=5, word_traning_length=40)
        print("padding {}".format(time.time() - starttime))
        print(model.is_words_in_batch(["aab", "bb", "ababababa", "dbdbd", "cccccd"]))

        a, samples = confidence_interval_many([model, lan()], random_word,
                                              epsilon=0.005)
        print(a)

        model.save_rnn("models2/" + str(i))


print("Begin")
# a = torch.tensor([[1,2],[2,1],[3,3]])
# a= a[:,1]
# a= a[:,-1]
main_train_RNNS()
# learn_dfa_and_compare_distance("models2")
