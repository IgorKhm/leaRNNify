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
    samples = None
    for folder in os.walk(dir):
        if folder[0] == dir:
            continue
        dfaOrigin = load_dfa_dot(folder[0] + r"\dfa.dot")
        ltsm = LSTMLanguageClasifier()
        ltsm.load_rnn(folder[0])

        # lstar_dfa = extract(ltsm, time_limit=300, initial_split_depth=20)
        teacher_pac = PACTeacher(ltsm)
        student_pac = DecisionTreeLearner(teacher_pac)
        teacher_pac.teach(student_pac, 300)

        a, samples = confidence_interval_many([dfaOrigin, ltsm, student_pac.dfa], random_word,
                                              epsilon=0.05)
        print(a)

    return


def main_train_RNNS():
    for round in range(1, 11):
        alphabet = "abcde"
        dfa_rand = random_dfa(alphabet, 10, 20, 5, 10)
        print(dfa_rand)
        teacher_exact = ExactTeacher(dfa_rand)
        student_exact = DecisionTreeLearner(teacher_exact)
        teacher_exact.teach(student_exact)
        dfa_rand = student_exact.dfa
        r = lan()
        print(r.is_word_in("bababababb"))
        print(dfa_rand)
        # dfa_rand.draw_nicely(name="_testing_stuff")
        save_dfa_as_part_of_model("models11/" + str(round), dfa_rand, True)
        #
        # starttime = time.time()
        # model1 = lstm()
        # model1.train_a_lstm(alphabet, dfa_rand.is_word_in, hidden_dim= int(2 * len(dfa_rand.states)), num_layers=int(len(dfa_rand.states)/10)+1,
        #                     embedding_dim=len(alphabet)*2,
        #                     num_of_exm_per_lenght=15000, batch_size=20, epoch=10,
        #                     word_traning_length=len(dfa_rand.states) + 5
        #                     )
        #
        # print("stright forward time: {}".format(time.time() - starttime))
        # starttime = time.time()
        # model = LSTMLanguageClasifier()
        # model.train_a_lstm(alphabet, dfa_rand.is_word_in, hidden_dim=int(3 * len(dfa_rand.states)), num_layers=int(len(dfa_rand.states)/10)+1,
        #                    embedding_dim=10,
        #                    num_of_exm_per_lenght=15000, batch_size=20, epoch=10,
        #                    word_traning_length=len(dfa_rand.states) + 5
        #                    )
        # print("packpad padding time: {}".format(time.time() - starttime))

        k, i, j = np.random.randint(1, 6), np.random.randint(1, 10), np.random.randint(1, 5)
        k, i, j = 2, 3, 1
        print('No multiplication batch = {}, hidden_dim = {}, num_layers = {}'.format(k, i, j))
        print('batch = {}, hidden_dim = {}, num_layers = {}'.format(10 * k, i * len(
            dfa_rand.states), int(len(dfa_rand.states) / 10) + j))
        starttime = time.time()
        model = LSTMLanguageClasifier()
        model.train_a_lstm(alphabet, dfa_rand.is_word_in,
                           hidden_dim=int(i * len(dfa_rand.states)),
                           num_layers=int(len(dfa_rand.states) / 10) + j,
                           embedding_dim=10,
                           num_of_exm_per_lenght=5000, batch_size=10 * k, epoch=10,
                           word_traning_length=len(dfa_rand.states) + 15
                           )
        print("time: {}".format(time.time() - starttime))

        #
        # a, _ = confidence_interval_many([model1, model, dfa_rand], random_word,
        #                                 epsilon=0.005)
        # print("m1 VS m = {}".format(a[0][1]))
        # print("m1 VS a = {}".format(a[0][2]))
        # print("m  VS a = {}".format(a[1][2]))
        model.save_rnn("models11/" + str(round), True)


print("Begin")
# a = torch.tensor([[1,2],[2,1],[3,3]])
# a= a[:,1]
# a= a[:,-1]
main_train_RNNS()
# learn_dfa_and_compare_distance("models2")
