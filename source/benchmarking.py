import csv
import datetime
import os
import time

import numpy as np

from dfa import DFA, random_dfa, dfa_intersection, save_dfa_as_part_of_model
from dfa_check import DFAChecker
from exact_teacher import ExactTeacher
from learner_decison_tree import DecisionTreeLearner
from lstar.Extraction import extract as extract_iclm
from modelPadding import LSTMLanguageClasifier
from pac_teacher import PACTeacher
from random_words import confidence_interval_many, random_word, confidence_interval_subset

FIELD_NAMES = ["alph_len",

               "dfa_inter_states", "dfa_inter_final",
               'dfa_spec_states', 'dfa_spec_final',
               'dfa_extract_specs_states', "dfa_extract_specs_final",
               "dfa_extract_states", "dfa_extract_final",
               "dfa_icml18_states", "dfa_icml18_final",

               "rnn_layers", "rnn_hidden_dim", "rnn_dataset_learning", "rnn_dataset_testing",
               "rnn_testing_acc", "rnn_val_acc", "rnn_time",

               "extraction_time_spec", "extraction_mistake_during",
               "extraction_time", "mistake_time_after", "extraction_mistake_after",
               "extraction_time_icml18",

               "dist_rnn_vs_inter", "dist_rnn_vs_extr", "dist_rnn_vs_extr_spec", "dist_rnn_vs_icml18",
               "dist_inter_vs_extr", "dist_inter_vs_extr_spec", "dist_inter_vs_icml18",

               "dist_specs_rnn", "dist_specs_extract", "dist_specs_extract_w_spec"]


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
        hidden_dim = len(dfa.states) * 3
    if num_layers == -1:
        num_layers = 2
    if embedding_dim == -1:
        embedding_dim = len(dfa.alphabet) * 2
    if num_of_exm_per_length == -1:
        num_of_exm_per_length = 15000
    if epoch == -1:
        epoch = 20
    if batch_size == -1:
        batch_size = 20
    if word_training_length == -1:
        word_training_length = len(dfa.states) + 5

    start_time = time.time()
    model = LSTMLanguageClasifier()
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


def learn_and_check(dfa: DFA, spec: [DFAChecker], benchmark, dir_name=None):
    rnn = learn_dfa(dfa, benchmark)

    extracted_dfas = check_rnn_acc_to_spec(rnn, spec, benchmark)
    if dir_name is not None:
        rnn.save_rnn(dir_name)
        for dfa, name in extracted_dfas:
            if isinstance(name, DFA):
                save_dfa_as_part_of_model(dir_name, dfa, name=name)
            # dfa_extract.draw_nicely(name="_dfa_figure", save_dir=dir_name)

    models = [dfa, rnn, extracted_dfas[0][0], extracted_dfas[1][0], extracted_dfas[2][0]]

    compute_distances(models, spec[0].specification, benchmark)


def check_rnn_acc_to_spec(rnn, spec, benchmark, timeout=900):
    teacher_pac = PACTeacher(rnn)
    student = DecisionTreeLearner(teacher_pac)

    print("Starting DFA extraction")
    ##################################################
    # Doing the model checking during a DFA extraction
    ###################################################
    print("Starting DFA extraction with model checking")
    start_time = time.time()
    counter = teacher_pac.check_and_teach(student, spec, timeout=timeout)
    benchmark.update({"extraction_time_spec": "{:.3}".format(time.time() - start_time)})
    dfa_extract_w_spec = student.dfa
    dfa_extract_w_spec = minimize_dfa(dfa_extract_w_spec)

    if counter is None:
        print("No mistakes found ==> DFA learned:")
        print(student.dfa)
        benchmark.update({"extraction_mistake_during": "",
                          "dfa_extract_specs_states": len(dfa_extract_w_spec.states),
                          "dfa_extract_specs_final": len(dfa_extract_w_spec.final_states)})
    else:
        print("Mistakes found ==> Counter example: {}".format(counter))
        benchmark.update({"extraction_mistake_during": counter[0],
                          "dfa_extract_specs_states": len(dfa_extract_w_spec.states),
                          "dfa_extract_specs_final": len(dfa_extract_w_spec.final_states)})

    ###################################################
    # Doing the model checking after a DFA extraction
    ###################################################
    print("Starting DFA extraction w/o model checking")
    start_time = time.time()
    student = DecisionTreeLearner(teacher_pac)
    teacher_pac.teach(student, timeout=timeout)
    benchmark.update({"extraction_time": "{:.3}".format(time.time() - start_time)})

    print("Model checking the extracted DFA")
    counter = student.dfa.is_language_not_subset_of(spec[0].specification)
    if counter is not None:
        if not rnn.is_word_in(counter):
            counter = None

    benchmark.update({"mistake_time_after": "{:.3}".format(time.time() - start_time)})

    dfa_extract = student.dfa
    if counter is None:
        print("No mistakes found ==> DFA learned:")
        print(student.dfa)
        benchmark.update({"extraction_mistake_after": "",
                          "dfa_extract_states": len(dfa_extract.states),
                          "dfa_extract_final": len(dfa_extract.final_states)})
    else:
        print("Mistakes found ==> Counter example: {}".format(counter))
        benchmark.update({"extraction_mistake_after": counter,
                          "dfa_extract_states": len(dfa_extract.states),
                          "dfa_extract_final": len(dfa_extract.final_states)})

    ###################################################
    # Doing DFA extraction acc. to icml18
    ###################################################
    print("Starting DFA extraction acc to iclm18")
    start_time = time.time()

    dfa_iclm18 = extract_iclm(rnn, time_limit=timeout, initial_split_depth=10)

    benchmark.update({"extraction_time_icml18": time.time() - start_time,
                      "dfa_icml18_states": len(dfa_iclm18.Q),
                      "dfa_icml18_final": len(dfa_iclm18.F)})

    print("Finished DFA extraction")

    return (dfa_extract_w_spec, "dfa_extract_W_spec"), \
           (dfa_extract, "dfa_extract"), \
           (dfa_iclm18, "dfa_icml18")


def compute_distances(models, dfa_spec, benchmark, epsilon=0.5, delta=0.01):
    print("Starting distance measuring")
    output, samples = confidence_interval_many(models, random_word, width=epsilon, confidence=delta)
    print("The confidence interval for epsilon = {} , delta = {}".format(delta, epsilon))
    print(output)
    # if len(models) == 3:
    #     print(" |----------------|----------------|----------------|-----------------|\n",
    #           "|                |  DFA original  |      RNN       |    DFA learned  |\n",
    #           "|----------------|----------------|----------------|-----------------|\n",
    #           "|  DFA original  |-----{:.4f}-----|-----{:.4f}-----|------{:.4f}-----|\n".format(output[0][0],
    #                                                                                             output[0][1],
    #                                                                                             output[0][2]),
    #           "|----------------|----------------|----------------|-----------------|\n",
    #           "|      RNN       |-----{:.4f}-----|-----{:.4f}-----|------{:.4f}-----|\n".format(output[1][0],
    #                                                                                             output[1][1],
    #                                                                                             output[1][2]),
    #           "|----------------|----------------|----------------|-----------------|\n",
    #           "|   DFA learned  |-----{:.4f}-----|-----{:.4f}-----|------{:.4f}-----|\n".format(output[2][0],
    #                                                                                             output[2][1],
    #                                                                                             output[2][2]),
    #           "|----------------|----------------|----------------|-----------------|\n")
    # else:
    #     print(" |----------------|----------------|----------------|\n",
    #           "|                |  DFA original  |      RNN       |\n",
    #           "|----------------|----------------|----------------|\n",
    #           "|  DFA original  |-----{:.4f}-----|-----{:.4f}-----|\n".format(output[0][0], output[0][1]),
    #           "|----------------|----------------|----------------|\n",
    #           "|      RNN       |-----{:.4f}-----|-----{:.4f}-----|\n".format(output[1][0], output[1][1]),
    #           "|----------------|----------------|----------------|\n")

    benchmark.update({"dist_rnn_vs_inter": "{:.4}".format(output[1][0]),
                      "dist_rnn_vs_extr_spec": "{:.4}".format(output[1][2]),
                      "dist_rnn_vs_extr": "{:.4}".format(output[1][3]),
                      "dist_rnn_vs_icml18": "{:.4}".format(output[1][4])})

    benchmark.update({"dist_inter_vs_extr_spec": "{:.4}".format(output[0][2]),
                      "dist_inter_vs_extr": "{:.4}".format(output[0][3]),
                      "dist_inter_vs_icml18": "{:.4}".format(output[0][4])})

    a, _ = confidence_interval_subset(models[1], dfa_spec, samples, epsilon, delta)
    b, _ = confidence_interval_subset(models[2], dfa_spec, samples, epsilon, delta)
    c, _ = confidence_interval_subset(models[3], dfa_spec, samples, epsilon, delta)
    benchmark.update(
        {"dist_specs_rnn": "{}".format(a),
         "dist_specs_extract_w_spec": "{}".format(b),
         "dist_specs_extract": "{}".format(c)})

    print("Finished distance measuring")


def rand_benchmark(save_dir=None):
    dfa_spec, dfa_inter = DFA(0, {0}, {0: {0: 0}}), DFA(0, {0}, {0: {0: 0}})

    full_alphabet = "abcdefghijklmnopqrstuvwxyz"

    alphabet = full_alphabet[0:np.random.randint(4, 10)]
    benchmark = {}
    benchmark.update({"alph_len": len(alphabet)})

    while len(dfa_inter.states) < 5 or len(dfa_spec.states) < 2 or (len(dfa_inter.states) > 35):
        dfa_rand1 = random_dfa(alphabet, min_states=10, max_states=20, min_final=2, max_final=10)
        dfa_rand2 = random_dfa(alphabet, min_states=5, max_states=7, min_final=4, max_final=5)

        dfa_inter = minimize_dfa(dfa_intersection(dfa_rand1, dfa_rand2))
        dfa_spec = minimize_dfa(dfa_rand2)

    benchmark.update({"dfa_inter_states": len(dfa_inter.states), "dfa_inter_final": len(dfa_inter.final_states),
                      "dfa_spec_states": len(dfa_spec.states), "dfa_spec_final": len(dfa_spec.final_states)})

    if save_dir is not None:
        save_dfa_as_part_of_model(save_dir, dfa_inter, name="dfa_intersection")
        dfa_inter.draw_nicely(name="intersection_dfa_figure", save_dir=save_dir)

        save_dfa_as_part_of_model(save_dir, dfa_spec, name="dfa_spec")
        dfa_spec.draw_nicely(name="spec_dfa_figure", save_dir=save_dir)

    print("DFA to learn {}".format(dfa_inter))
    print("Spec to learn {}".format(dfa_spec))

    learn_and_check(dfa_inter, [DFAChecker(dfa_spec)], benchmark, save_dir)

    return benchmark


def run_rand_benchmarks(num_of_bench=10, save_dir=None):
    if save_dir is None:
        save_dir = "../models/random_bench_{}".format(datetime.datetime.now().strftime("%d-%b-%Y_%H-%M-%S"))
        os.makedirs(save_dir)

    write_csv_header(save_dir + "/test.csv")
    for num in range(1, num_of_bench + 1):
        print("Running benchmark {}/{}:".format(num, num_of_bench))
        benchmark = rand_benchmark(save_dir + "/" + str(num))
        print("Summary for the {}th benchmark".format(num))
        print(benchmark)
        write_line_csv(save_dir + "/test.csv", benchmark)


def learn_multiple_times(dfa, dir_save=None):
    for hidden_dim, num_layers in ((20, 2), (50, 5), (100, 10), (200, 20), (500, 50)):
        benchmarks = {}
        lstm = learn_dfa(dfa, benchmarks,
                         hidden_dim=hidden_dim,
                         num_layers=hidden_dim,
                         num_of_exm_per_length=20000,
                         word_training_length=len(dfa.states) + 10)
        print(benchmarks)
        if dir_save is not None:
            lstm.save_rnn(dir_save + "/" + "l-{}__h-{}".format(num_layers, hidden_dim))


def run_multiple_spec_on_ltsm(ltsm, spec_dfas, messages):
    i = 1
    benchmark = {}
    check_rnn_acc_to_spec(ltsm, [DFAChecker(spec_dfas[5], is_super_set=False)], benchmark,
                          timeout=1800)

    for dfa, message in zip(spec_dfas, messages):
        print(message)
        check_rnn_acc_to_spec(ltsm, [DFAChecker(dfa)], benchmark,
                              timeout=1800)
        print(benchmark)
