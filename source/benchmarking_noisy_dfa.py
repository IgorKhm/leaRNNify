import csv
import datetime
import os
import time

import numpy as np

from dfa import DFA, random_dfa, dfa_intersection, save_dfa_as_part_of_model, DFANoisy
from dfa_check import DFAChecker
from exact_teacher import ExactTeacher
from learner_decison_tree import DecisionTreeLearner
from lstar.Extraction import extract as extract_iclm
from modelPadding import RNNLanguageClasifier
from pac_teacher import PACTeacher
from random_words import confidence_interval_many, random_word, confidence_interval_subset

FIELD_NAMES = ["alph_len",

               "dfa_states", "dfa_final",
               "dfa_extract_states", "dfa_extract_final",

               "extraction_time",

               "dist_dfa_vs_noisy", "dist_dfa_vs_extr", "dist_noisy_vs_extr"]


def write_csv_header(filename):
    with open(filename, mode='a') as employee_file:
        writer = csv.DictWriter(employee_file, fieldnames=FIELD_NAMES)
        writer.writeheader()


def write_line_csv(filename, benchmark):
    with open(filename, mode='a') as benchmark_summary:
        writer = csv.DictWriter(benchmark_summary, fieldnames=FIELD_NAMES)
        writer.writerow(benchmark)


def minimize_dfa(dfa: DFA) -> DFA:
    teacher_pac = ExactTeacher(dfa)
    student = DecisionTreeLearner(teacher_pac)
    teacher_pac.teach(student)
    return student.dfa


def learn_dfa(dfa: DFA, benchmark, hidden_dim=-1, num_layers=-1, embedding_dim=-1, batch_size=-1,
              epoch=-1, num_of_exm_per_length=-1, word_training_length=-1):
    if hidden_dim == -1:
        hidden_dim = len(dfa.states) * 6
    if num_layers == -1:
        num_layers = 3
    if embedding_dim == -1:
        embedding_dim = len(dfa.alphabet) * 2
    if num_of_exm_per_length == -1:
        num_of_exm_per_length = 15000
    if epoch == -1:
        epoch = 10
    if batch_size == -1:
        batch_size = 20
    if word_training_length == -1:
        word_training_length = len(dfa.states) + 5

    start_time = time.time()
    model = RNNLanguageClasifier()
    model.train_a_lstm(dfa.alphabet, dfa.is_word_in,
                       hidden_dim=hidden_dim,
                       num_layers=num_layers,
                       embedding_dim=embedding_dim,
                       batch_size=batch_size,
                       epoch=epoch,
                       num_of_exm_per_lenght=num_of_exm_per_length,
                       word_traning_length=word_training_length
                       )

    benchmark.update({"rnn_time": "{:.3}".format(time.time() - start_time),
                      "rnn_hidden_dim": hidden_dim,
                      "rnn_layers": num_layers,
                      "rnn_testing_acc": "{:.3}".format(model.test_acc),
                      "rnn_val_acc": "{:.3}".format(model.val_acc),
                      "rnn_dataset_learning": model.num_of_train,
                      "rnn_dataset_testing": model.num_of_test})

    print("time: {}".format(time.time() - start_time))
    return model


def extract_mesaure(dfa: DFA, benchmark, dir_name=None):
    dfa_noisy = DFANoisy(dfa.init_state, dfa.final_states, dfa.transitions, mistake_prob=0.001)
    extracted_dfa = check_rnn_acc_to_spec(dfa_noisy, benchmark, timeout=900)
    if dir_name is not None:
        save_dfa_as_part_of_model(dir_name, extracted_dfa, name="extracted_dfa")

    models = [dfa, dfa_noisy, extracted_dfa]

    compute_distances(models, benchmark)


def check_rnn_acc_to_spec(dfa, benchmark, timeout=900):
    teacher_pac = PACTeacher(dfa)
    student = DecisionTreeLearner(teacher_pac)

    print("Starting DFA extraction")
    ###################################################
    # Doing the model checking after a DFA extraction
    ###################################################
    start_time = time.time()
    student = DecisionTreeLearner(teacher_pac)
    teacher_pac.teach(student, timeout=timeout)
    benchmark.update({"extraction_time": "{:.3}".format(time.time() - start_time)})

    dfa_extract = minimize_dfa(student.dfa)
    print(student.dfa)
    benchmark.update({"dfa_extract_states": len(dfa_extract.states),
                      "dfa_extract_final": len(dfa_extract.final_states)})

    return dfa_extract


def compute_distances(models, benchmark, epsilon=0.005, delta=0.001):
    print("Starting distance measuring")
    output, samples = confidence_interval_many(models, random_word, width=epsilon, confidence=delta)
    print("The confidence interval for epsilon = {} , delta = {}".format(delta, epsilon))
    print(output)

    benchmark.update({"dist_dfa_vs_noisy": "{}".format(output[0][1]),
                      "dist_dfa_vs_extr": "{}".format(output[0][2]),
                      "dist_noisy_vs_extr": "{}".format(output[1][2])})

    print("Finished distance measuring")


def rand_benchmark(save_dir=None):
    full_alphabet = "abcdefghijklmnopqrstuvwxyz"

    alphabet = full_alphabet[0:np.random.randint(4, 10)]
    benchmark = {}
    benchmark.update({"alph_len": len(alphabet)})

    max_final = np.random.randint(6, 40)

    dfa_rand = random_dfa(alphabet, min_states=max_final + 1, max_states=50, min_final=5, max_final=max_final)
    dfa = minimize_dfa(dfa_rand)

    benchmark.update({"dfa_states": len(dfa.states), "dfa_final": len(dfa.final_states)})

    if save_dir is not None:
        save_dfa_as_part_of_model(save_dir, dfa, name="dfa")

    print("DFA to learn {}".format(dfa))

    extract_mesaure(dfa, benchmark, save_dir)

    return benchmark


def run_rand_benchmarks_noisy_dfa(num_of_bench=10, save_dir=None):
    if save_dir is None:
        save_dir = "../models/random_bench_noisy_dfa_{}".format(datetime.datetime.now().strftime("%d-%b-%Y_%H-%M-%S"))
        os.makedirs(save_dir)

    write_csv_header(save_dir + "/test.csv")
    for num in range(1, num_of_bench + 1):
        print("Running benchmark {}/{}:".format(num, num_of_bench))
        benchmark = rand_benchmark(save_dir + "/" + str(num))
        print("Summary for the {}th benchmark".format(num))
        print(benchmark)
        write_line_csv(save_dir + "/test.csv", benchmark)
