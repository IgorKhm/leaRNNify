import csv
import datetime
import os
import time

from dfa import DFA, random_dfa, dfa_intersection, save_dfa_as_part_of_model
from dfa_check import DFAChecker
from exact_teacher import ExactTeacher
from learner_decison_tree import DecisionTreeLearner
from modelPadding import LSTMLanguageClasifier
from pac_teacher import PACTeacher
from random_words import confidence_interval_many, random_word, confidence_interval_subset

FIELD_NAMES = ["alph_len",

               "dfa_inter_states", "dfa_spec_states",
               "dfa_extract_states", "dfa_inter_final",
               "dfa_spec_final", "dfa_extract_final",

               "lstm_layers", "lstm_hidden_dim", "lstm_dataset_learning", "lstm_dataset_testing",
               "lstm_testing_acc", "lstm_val_acc", "lstm_time",

               "extraction_time", "extraction_mistake",

               "dist_lstm_vs_inter", "dist_lstm_vs_extr", "dist_extr_vs_inter",
               "dist_lstm_specs", "dist_extract_specs"]


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

    benchmark.update({"lstm_time": "{:.3}".format(time.time() - start_time),
                      "lstm_hidden_dim": hidden_dim,
                      "lstm_layers": num_layers,
                      "lstm_testing_acc": "{:.3}".format(model.test_acc),
                      "lstm_val_acc": "{:.3}".format(model.val_acc),
                      "lstm_dataset_learning": model.num_of_train,
                      "lstm_dataset_testing": model.num_of_test})

    print("time: {}".format(time.time() - start_time))
    return model


def learn_and_check(dfa: DFA, spec: [DFAChecker], benchmark, dir_name=None):
    rnn = learn_dfa(dfa, benchmark)

    dfa_extract = check_rnn_acc_to_spec_and_original_dfa(rnn, spec, benchmark)

    if dir_name is not None:
        rnn.save_rnn(dir_name)
        save_dfa_as_part_of_model(dir_name, dfa_extract, name="extract_dfa")
        dfa_extract.draw_nicely(name="extract_dfa_figure", save_dir=dir_name)

    compute_distances(dfa, rnn, dfa_extract, spec[0].specificatio, benchmark)


def check_rnn_acc_to_spec_and_original_dfa(rnn, spec, benchmark):
    teacher_pac = PACTeacher(rnn)
    student = DecisionTreeLearner(teacher_pac)

    print("Starting DFA extraction")
    start_time = time.time()
    counter = teacher_pac.check_and_teach(student, spec)
    benchmark.update({"extraction_time": "{:.3}".format(time.time() - start_time)})

    if counter is None:
        print("No mistakes found ==> DFA learned:")
        print(student.dfa)
        benchmark.update({"extraction_mistake": "",
                          "dfa_spec_final": len(student.dfa.states),
                          "dfa_extract_final": len(student.dfa.final_states)})
    else:
        print("Mistakes found ==> Counter example: {}".format(counter))
        benchmark.update({"extraction_mistake": counter[0],
                          "dfa_spec_final": len(student.dfa.states),
                          "dfa_extract_final": len(student.dfa.final_states)})

    print("Finished DFA extraction")
    return student.dfa


def compute_distances(dfa_original, rnn, dfa_learned, dfa_spec, benchmark):
    models = [dfa_original, rnn, dfa_learned]
    print("Starting distance measuring")
    epsilon = 0.005
    delta = 0.001
    output, samples = confidence_interval_many(models, random_word, epsilon=epsilon, delta=delta)
    print("The confidence interval for epsilon = {} , delta = {}".format(delta, epsilon))
    if len(models) == 3:
        print(" |----------------|----------------|----------------|-----------------|\n",
              "|                |  DFA original  |      RNN       |    DFA learned  |\n",
              "|----------------|----------------|----------------|-----------------|\n",
              "|  DFA original  |-----{:.4f}-----|-----{:.4f}-----|------{:.4f}-----|\n".format(output[0][0],
                                                                                                output[0][1],
                                                                                                output[0][2]),
              "|----------------|----------------|----------------|-----------------|\n",
              "|      RNN       |-----{:.4f}-----|-----{:.4f}-----|------{:.4f}-----|\n".format(output[1][0],
                                                                                                output[1][1],
                                                                                                output[1][2]),
              "|----------------|----------------|----------------|-----------------|\n",
              "|   DFA learned  |-----{:.4f}-----|-----{:.4f}-----|------{:.4f}-----|\n".format(output[2][0],
                                                                                                output[2][1],
                                                                                                output[2][2]),
              "|----------------|----------------|----------------|-----------------|\n")
    else:
        print(" |----------------|----------------|----------------|\n",
              "|                |  DFA original  |      RNN       |\n",
              "|----------------|----------------|----------------|\n",
              "|  DFA original  |-----{:.4f}-----|-----{:.4f}-----|\n".format(output[0][0], output[0][1]),
              "|----------------|----------------|----------------|\n",
              "|      RNN       |-----{:.4f}-----|-----{:.4f}-----|\n".format(output[1][0], output[1][1]),
              "|----------------|----------------|----------------|\n")

    benchmark.update({"dist_lstm_vs_inter": "{:.4}".format(output[0][1]),
                      "dist_lstm_vs_extr": "{:.4}".format(output[0][2]),
                      "dist_extr_vs_inter": "{:.4}".format(output[2][1])})

    a, _ = confidence_interval_subset(rnn, dfa_spec, samples, epsilon, delta)
    b, _ = confidence_interval_subset(dfa_learned, dfa_spec, samples, epsilon, delta)
    benchmark.update(
        {"dist_lstm_specs": "{}".format(a),
         "dist_extract_specs": "{}".format(b)})

    print("Finished distance measuring")


def rand_benchmark(save_dir=None):
    dfa_inter = DFA(0, {0}, {0: {0: 0}})

    alphabet = "abcde"
    benchmark = {}
    benchmark.update({"alph_len": len(alphabet)})

    while len(dfa_inter.states) < 5:
        dfa_rand1 = random_dfa(alphabet, min_states=5, max_states=10, min_final=2, max_final=5)
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
