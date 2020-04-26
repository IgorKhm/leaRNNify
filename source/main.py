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
from model import LSTMLanguageClasifier, test_rnn

from pac_teacher import PACTeacher
from random_words import confidence_interval, random_word, confidence_interval_many

from dfa import save_dfa_dot, save_dfa_as_part_of_model
from exact_teacher import ExactTeacher
from lstar.Extraction import extract


def main():
    # a = input()
    # choose_word(a, 3)

    # dfa = DFA(1, {1}, {1: {"a": 2, "b": 1, "c": 1},
    #                    2: {"a": 3, "b": 1, "c": 3},
    #                    3: {"a": 3, "b": 3, "c": 3}})
    #
    # dfa2 = DFA(1, {1, 4}, {1: {"a": 2, "b": 1, "c": 1},
    #                        2: {"a": 3, "b": 1, "c": 3},
    #                        3: {"a": 3, "b": 3, "c": 4},
    #                        4: {"a": 3, "b": 3, "c": 4}})
    #
    # dfaKV = DFA(1, {4}, {1: {"0": 1, "1": 2},
    #                      2: {"0": 2, "1": 3},
    #                      3: {"0": 3, "1": 4},
    #                      4: {"0": 4, "1": 1}})
    #
    # dfabug = DFA(0, {2, 3}, {0: {"0": 7, "1": 3},
    #                          1: {"0": 6, "1": 2},
    #                          2: {"0": 0, "1": 5},
    #                          3: {"0": 2, "1": 2},
    #                          4: {"0": 0, "1": 3},
    #                          5: {"0": 0, "1": 4},
    #                          6: {"0": 5, "1": 2},
    #                          7: {"0": 3, "1": 7}})
    #
    # a = False
    # b = False
    # c = not a ^ b

    # dfa.draw_nicely()
    #
    # b = random_word_by_letter(["a", "b", "c"])
    #
    # while True:
    #     print(next(b))

    #
    # print(a[0:len(a) - 1])
    # # # b= DecisionTreeLearner()
    # a = PACTeacher(dfabug)
    #
    # # # a.membership_query("abc")
    # #
    # # s = ""
    # # s2 = "a"
    # # print(s + s2)
    # #
    # student = DecisionTreeLearner(a)

    # print("yoy")
    #
    # a.model.draw_nicely(name="1")

    # while True:
    #     # student.dfa.draw_nicely(name="2")
    #     counter = a.equivalence_query(student.dfa)
    #     if counter is None:
    #         break
    #     if student.dfa.is_word_in(counter) == a.model.is_word_in(counter) :
    #         print("?????")
    #     print(len(student.dfa.states))
    #     if len(student.dfa.states) == 6:
    #         print("")
    #     student.new_counterexample(counter)

    # student.dfa.draw_nicely()

    # if student.dfa == dfa:
    #     print("we are going places")
    #
    # if dfa != dfa2:
    #     print("good1")
    #
    # print(dfa.equivalence_with_counterexample(dfa2))
    #
    # if dfa.equivalence_with_counterexample(dfa) is None:
    #     print("good3")

    # #
    # while True:
    #     dfa3 = random_dfa(["a", "b", "c", "d"])
    #     dfa3.draw_nicely(name="1")
    #
    #     teacher = ExactTeacher(dfa3)
    #     # a.membership_query("abc")
    #
    #     student = DecisionTreeLearner(teacher)
    #     student.dfa.draw_nicely(name="2")
    #     counter = ""
    #     number_states = len(student.dfa.states)
    #     while True:
    #         counter2 = teacher.equivalence_query(student.dfa)
    #         if counter2 == counter:
    #             print("1?")
    #         counter = counter2
    #         if counter is None:
    #             break
    #         print(counter)
    #         student.new_counterexample(counter)
    #         if len(student.dfa.states) == number_states:
    #             print("not changing")
    #         number_states = len(student.dfa.states)
    #         # student.dfa.draw_nicely(name="2")
    #
    #     student.dfa.draw_nicely(name="2")
    #     if dfa3 != student.dfa:
    #         print("not equal")
    #     if len(student.dfa.states) > 50:
    #         break

    #

    def target(w):
        if len(w) == 0:
            return True
        return w[0] == w[-1]

    #

    #
    # dfa_rand = random_dfa(["a", "b", "c"], min_states=10, max_states=20, min_final=1,
    #                       max_final=3)
    # dfa_rand.draw_nicely(name="_rand")
    #
    # target = dfa_rand.is_word_in
    #
    # alphabet = "abc"
    #
    #
    # train_set = make_train_set_for_target(target, alphabet)
    # rnn = RNNClassifier(alphabet, num_layers=1, hidden_dim=10, RNNClass=LSTMNetwork)
    # mixed_curriculum_train(rnn, train_set, stop_threshold=0.005)
    #
    # print("done learning")
    # #
    # while True:
    #     # try:
    #     teacher_pac = PACTeacher(rnn)
    #     student_pac = DecisionTreeLearner(teacher_pac)
    #     teacher_pac.teach(student_pac)
    #     print("dfa learned")
    #     student_pac.dfa.draw_nicely(name="rnn")
    #     break
    #     # except:
    #     print("Asd")
    #
    # if student_pac.dfa == dfa_rand:
    #     print("cool")
    # else:
    #     print("oh well")

    def word_to_nparray(word, letter_to_num, length):
        array = np.zeros(length)
        for i in range(len(word)):
            array[length - i - 1] = letter_to_num[word[-i - 1]]

        return array

    #
    #
    alphabet = "abcdef"
    # dfa_rand = random_dfa(alphabet, 10, 20, 2, 8)
    # # # teacher_exact = ExactTeacher(dfa_rand)
    # # # # student_exact = DecisionTreeLearner(teacher_exact)
    # # # # teacher_exact.teach(student_exact)
    # # # # dfa_rand = student_exact.dfa
    # # # # # dfa_rand.draw_nicely(name="_rand")
    # # # # # # #
    # # save_dfa_dot("dfa4", dfa_rand)
    # model = LSTMLanguageClasifier()
    # test = model.train_a_lstm(alphabet, dfa_rand.is_word_in,hidden_dim=len(random_dfa.states))
    # model.save_rnn("test4")
    # # # load_dfa("toyDFA.doy")
    #
    # teacher_pac = PACTeacher(model)
    # student_pac = DecisionTreeLearner(teacher_pac)

    # teacher_pac.teach(student_pac)
    # print("dfa learned")
    # student_pac.dfa.draw_nicely(name="_rnn")
    # #
    #
    # b1 = model.is_word_in("a")
    # b2 = model.is_word_in("babab")
    # b3 = model.is_word_in("acabbcab")
    # b4 = model.is_word_in("bbbbbbbbbbba")
    # b5 = model.is_word_in("cbbbbbbc")
    # for i in range(10000):
    #     if b1 != model.is_word_in("a"):
    #         print("ahhhhhh")
    #     if b2 != model.is_word_in("babab"):
    #         print("ahhhhhh")
    #     if b3 != model.is_word_in("acabbcab"):
    #         print("ahhhhhh")
    #     if b4 != model.is_word_in("bbbbbbbbbbba"):
    #         print("ahhhhhh")
    #     if b5 != model.is_word_in("cbbbbbbc"):
    #         print("ahhhhhh")

    model2 = LSTMLanguageClasifier()
    model2.load_rnn("test4")
    model2.is_word_in("abaaba")

    # print(model2.is_word_in("abaaba"))
    # print(model2.is_word_in("aaaaaabac"))
    # model2.is_word_in("abaaba")

    #
    # for i in range(10000):
    #     print(i)
    #     if b1 != model2.is_word_in("a"):
    #         print("ahhhhhh")
    #     if b2 != model2.is_word_in("babab"):
    #         print("ahhhhhh")
    #     if b3 != model2.is_word_in("acabbcab"):
    #         print("ahhhhhh")
    #     if b4 != model2.is_word_in("bbbbbbbbbbba"):
    #         print("ahhhhhh")
    #     if b5 != model2.is_word_in("cbbbbbbc"):
    #         print("ahhhhhh")

    # print(model2.is_word_in("cd"))
    # print(model2.is_word_in("abbbba"))
    # print(target("a"))
    # test_rnn(model2._ltsm, test, 20, torch.device("cpu"))
    class Lang:
        def __init__(self, is_word):
            self.is_word_in = is_word

    lan = Lang(target)
    # #
    teacher_pac = PACTeacher(model2)
    student_pac = DecisionTreeLearner(teacher_pac)
    #
    # samples = set()
    # while len(samples) < 1000:
    #     w = random_word("abcd")
    #     if w not in samples:
    #         samples.add(w)
    # s = list(samples)
    # print("k")
    # a = model2.is_words_in_batch(["abab","abbb","bbbb","aaaa","baaa"])
    # s.pop(0)
    # a= model2.is_words_in_batch(samples)
    # print(a)
    # return

    # samples = set()
    # while len(samples) < 100:
    #     w = random_word("abcd")
    #     if w not in samples:
    #         samples.add(w)
    # print("words genrated")
    #
    # print("with nograd and eval on layers(in one batch):")
    #
    # t = time.time()
    # pr = cProfile.Profile()
    # pr.enable()
    #
    # model2._ltsm.lstm.eval()
    # model2._ltsm.fc.eval()
    # model2._ltsm.embedding.eval()
    # samplesL = list(samples)
    # with torch.no_grad():
    #     for i in range(1):
    #         model2.is_words_in_batch(samplesL[i * int(len(samplesL) / 1):(i + 1) * int(len(samplesL) / 1)]) > 0.5
    #
    #     # model2.is_words_in_batch(smapleL[int(len(smapleL) / 4): 2 * int(len(smapleL) / 3)]) > 0.5
    #     # model2.is_words_in_batch(smapleL[(2 * int(len(smapleL) / 3)): len(smapleL)])
    #
    # pr.disable()
    # pr.print_stats()
    # print(time.time() - t)

    # t = time.time()
    # pr = cProfile.Profile()
    # pr.enable()
    #
    # model2._ltsm.lstm.eval()
    # model2._ltsm.fc.eval()
    # model2._ltsm.embedding.eval()
    # samplesL = list(samples)
    # with torch.no_grad():
    #
    #     n = int(len(samplesL) / 50)
    #     for i in range(n):
    #         model2.is_words_in_batch(samplesL[i * int(len(samplesL) / n):(i + 1) * int(len(samplesL) / n)]) > 0.5
    #
    #     # model2.is_words_in_batch(smapleL[int(len(smapleL) / 4): 2 * int(len(smapleL) / 3)]) > 0.5
    #     # model2.is_words_in_batch(smapleL[(2 * int(len(smapleL) / 3)): len(smapleL)])
    #
    # pr.disable()
    # pr.print_stats()
    # print(time.time() - t)
    #
    #
    # print("with nograd:")
    # t = time.time()
    # with torch.no_grad():
    #     for w in samples:
    #         result = model2.is_word_in(w)
    #         for i in range(100):
    #             if result != model2.is_word_in_test(w,len(w)+i):
    #                 print("yeah.......")
    #                 print(w)
    #                 print(i)
    # print(time.time() - t)
    #
    # print("without nograd:")
    # t = time.time()
    # for w in samples:
    #     model2.is_word_in(w)
    # print(time.time() - t)
    #
    # print("with nograd and eval on layers:")
    # t = time.time()
    # model2._ltsm.lstm.eval()
    # model2._ltsm.fc.eval()
    # model2._ltsm.embedding.eval()
    # with torch.no_grad():
    #     for w in samples:
    #         model2.is_word_in(w)
    # print(time.time() - t)
    #
    # return
    pr = cProfile.Profile()
    pr.enable()
    teacher_pac.teach(student_pac)
    pr.disable()
    pr.print_stats()
    return
    # print("dfa learned")
    # student_pac.dfa.draw_nicely(name="_rnn")
    # # except:
    # print("Asd")

    rvs, samples = confidence_interval(model2, student_pac.dfa, random_word)
    rvl, _ = confidence_interval(model2, dfa_rand, random_word, samples=samples)
    svl, _ = confidence_interval(student_pac.dfa, dfa_rand, random_word, samples=samples)

    print("Confidence rnn vs studenf {}".format(rvs))
    print("Confidence rnn vs lan {}".format(rvl))
    print("Confidence student vs lan {}".format(svl))
    # if student_pac.dfa == dfa_rand:
    #     print("cool")
    # else:
    #     print("oh well")

    #
    #

    return


def target(w):
    return w[0] == w[-1]


def main_train_RNNS():
    for i in range(5,10):
        alphabet = "abcdef"
        dfa_rand = random_dfa(alphabet, 100, 120, 1, 39)
        print(dfa_rand)
        teacher_exact = ExactTeacher(dfa_rand)
        student_exact = DecisionTreeLearner(teacher_exact)
        teacher_exact.teach(student_exact)
        dfa_rand = student_exact.dfa
        print(dfa_rand)
        # dfa_rand.draw_nicely(name="_testing_stuff")
        save_dfa_as_part_of_model("models2/" + str(i), dfa_rand)
        model = LSTMLanguageClasifier()
        test = model.train_a_lstm(alphabet, dfa_rand.is_word_in, hidden_dim=len(dfa_rand.states),
                                  num_of_exm_per_lenght=20000, batch_size=50)
        print("here")
        # test_rnn(model, test, 30, model._ltsm.device)
        model.save_rnn("models2/" + str(i))


def old_main():
    ###################################################################################################
    ###################################################################################################
    ###################################################################################################
    ######################## NEXT COMES THE NN CODE DONT DELETE!!!!!! #################################
    ###################################################################################################
    ###################################################################################################
    ###################################################################################################

    def from_array_to_word(int2char, array):
        # a = word_to_nparray(w, char2int, 30)
        # print(a)
        word = ""
        for i in array:
            word = word + int2char[i]
        return word

    def make_training_sets(alphabet, target, num_of_exm_per_lenght=200):
        int2char = ({i + 1: alphabet[i] for i in range(len(alphabet))})
        int2char.update({0: ""})
        char2int = {alphabet[i]: i + 1 for i in range(len(alphabet))}
        char2int.update({"": 0})
        #
        # a  = np.random.randint(5, size=(2, 5))
        # test= (np.random.randint(1, len(alphabet)+1, size=(num_of_exm_per_lenght, 15)))
        words_list = []

        lengths = list(range(1, 15)) + list(range(20, 50, 5))
        for length in lengths:
            new_list = np.unique(np.random.randint(1, len(alphabet) + 1, size=(num_of_exm_per_lenght, length)), axis=0)
            new_list = [np.pad(w, (60 - length, 0)) for w in new_list]
            print("length: " + str(length) + ", new list: " + str(len(new_list)))
            if words_list is None:
                words_list = new_list
            else:
                words_list.extend(new_list)
        label_list = [target(from_array_to_word(int2char, w)) for w in words_list]

        print("here")
        # chooses 10% words for test batch
        test_words, test_label = [], []
        for _ in range(int(len(words_list) / 10)):
            i = np.random.randint(0, len(words_list))
            test_words.append(words_list[i])
            test_label.append(label_list[i])
            del words_list[i]
            del label_list[i]
        print("here")
        # split the rest between validation and learning
        val_words, val_label = [], []
        for _ in range(int(len(words_list) / 2)):
            i = np.random.randint(0, len(words_list))
            val_words.append(words_list[i])
            val_label.append(label_list[i])
            del words_list[i]
            del label_list[i]

        train_words, train_label = words_list, label_list

        return test_words[0:int(len(test_words) / 100) * 100], test_label[
                                                               0:int(len(test_label) / 100) * 100], val_words[
                                                                                                    0:int(
                                                                                                        len(
                                                                                                            val_words) / 100) * 100], val_label[
                                                                                                                                      0:int(
                                                                                                                                          len(
                                                                                                                                              val_label) / 100) * 100], train_words[
                                                                                                                                                                        0:int(
                                                                                                                                                                            len(
                                                                                                                                                                                train_words) / 100) * 100], train_label[
                                                                                                                                                                                                            0:int(
                                                                                                                                                                                                                len(
                                                                                                                                                                                                                    train_label) / 100) * 100]

        nums_of_letters = len(alphabet)
        letter = np.random.randint(0, nums_of_letters)
        word = ""
        while letter != 0:
            word = word + alphabet[letter]
            letter = np.random.randint(0, nums_of_letters)
        return word

    test_words, test_labels, val_words, val_labels, train_words, train_labels = make_training_sets(alphabet, target,
                                                                                                   50000)

    print(len(train_words))

    train_data = TensorDataset(torch.from_numpy(np.array(train_words)), torch.from_numpy(np.array(train_labels)))
    val_data = TensorDataset(torch.from_numpy(np.array(val_words)), torch.from_numpy(np.array(val_labels)))
    test_data = TensorDataset(torch.from_numpy(np.array(test_words)), torch.from_numpy(np.array(test_labels)))

    batch_size = 20

    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
    val_loader = DataLoader(val_data, shuffle=True, batch_size=batch_size)
    test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size)

    # torch.cuda.is_available() checks and returns a Boolean True if a GPU is available, else it'll return False
    is_cuda = torch.cuda.is_available()

    # If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
    if is_cuda:
        device = torch.device("cuda")
        print("GPU is available")
    else:
        device = torch.device("cpu")
        print("GPU not available, CPU used")

    class SentimentNet(nn.Module):
        def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers, drop_prob=0.5):
            super(SentimentNet, self).__init__()
            self.output_size = output_size
            self.n_layers = n_layers
            self.hidden_dim = hidden_dim

            self.embedding = nn.Embedding(vocab_size, embedding_dim)
            self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=drop_prob, batch_first=True)
            self.dropout = nn.Dropout(0.2)
            self.fc = nn.Linear(hidden_dim, output_size)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x, hidden):
            batch_size = x.size(0)
            x = x.long()
            embeds = self.embedding(x)
            lstm_out, hidden = self.lstm(embeds, hidden)
            lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)

            out = self.dropout(lstm_out)
            out = self.fc(out)
            out = self.sigmoid(out)

            out = out.view(batch_size, -1)
            out = out[:, -1]
            return out, hidden

        def init_hidden(self, batch_size):
            weight = next(self.parameters()).data
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device),
                      weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device))
            return hidden

    alphabet = "abcd"
    vocab_size = len(alphabet) + 1
    output_size = 1
    embedding_dim = 10
    hidden_dim = 10
    n_layers = 2

    model = SentimentNet(vocab_size, output_size, embedding_dim, hidden_dim, n_layers)
    model.to(device)
    print(model)

    lr = 0.005
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    epochs = 10
    counter = 0
    print_every = 1000
    clip = 5
    valid_loss_min = np.Inf

    model.train()
    for i in range(epochs):
        h = model.init_hidden(batch_size)

        for inputs, labels in train_loader:
            counter += 1
            h = tuple([e.data for e in h])
            inputs, labels = inputs.to(device), labels.to(device)
            model.zero_grad()
            output, h = model(inputs, h)
            loss = criterion(output.squeeze(), labels.float())
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()

            if counter % print_every == 0:
                val_h = model.init_hidden(batch_size)
                val_losses = []
                model.eval()
                for inp, lab in val_loader:
                    val_h = tuple([each.data for each in val_h])
                    inp, lab = inp.to(device), lab.to(device)
                    out, val_h = model(inp, val_h)
                    val_loss = criterion(out.squeeze(), lab.float())
                    val_losses.append(val_loss.item())

                model.train()
                print("Epoch: {}/{}...".format(i + 1, epochs),
                      "Step: {}...".format(counter),
                      "Loss: {:.6f}...".format(loss.item()),
                      "Val Loss: {:.6f}".format(np.mean(val_losses)))
                if np.mean(val_losses) <= valid_loss_min:
                    torch.save(model.state_dict(), './state_dict.pt')
                    print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min,
                                                                                                    np.mean(
                                                                                                        val_losses)))
                    valid_loss_min = np.mean(val_losses)

    print("done")

    # Loading the best model
    model.load_state_dict(torch.load('./state_dict.pt'))

    test_losses = []
    num_correct = 0
    h = model.init_hidden(batch_size)

    model.eval()
    for inputs, labels in test_loader:
        h = tuple([each.data for each in h])
        inputs, labels = inputs.to(device), labels.to(device)
        output, h = model(inputs, h)
        test_loss = criterion(output.squeeze(), labels.float())
        test_losses.append(test_loss.item())
        pred = torch.round(output.squeeze())  # rounds the output to 0/1
        correct_tensor = pred.eq(labels.float().view_as(pred))
        correct = np.squeeze(correct_tensor.cpu().numpy())
        num_correct += np.sum(correct)

    print("Test loss: {:.3f}".format(np.mean(test_losses)))
    test_acc = num_correct / len(test_loader.dataset)
    print("Test accuracy: {:.3f}%".format(test_acc * 100))

    word = "baababababababacbabcabcbacbabcabababbaabababcbacbabcabcabbaabaaccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccb"
    char2int = {alphabet[i]: i + 1 for i in range(len(alphabet))}
    # w = torch.from_numpy(np.array([[char2int[letter] for letter in word]]))
    h = model.init_hidden(1)
    for l in word:
        t = torch.from_numpy(np.array([[char2int[l]]]))
        output, h = model(t, h)
        print(output)

    print(output > 0.5)


def learn_dfa_and_compare_distance(dir):
    for folder in os.walk(dir):
        if folder[0] == dir:
            continue
        dfaOrigin = load_dfa_dot(folder[0] + r"\dfa.dot")
        ltsm = LSTMLanguageClasifier()
        ltsm.load_rnn(folder[0])

        # lstar_dfa = extract(ltsm,time_limit = 600, initial_split_depth=20)
        teacher_pac = PACTeacher(ltsm)
        student_pac = DecisionTreeLearner(teacher_pac)
        # teacher_pac.teach(student_pac,600)

        a, samples = confidence_interval_many([dfaOrigin, ltsm, student_pac], random_word,
                                              epsilon=0.005)
        print(a)

    return

    model2 = LSTMLanguageClasifier()
    model2.load_rnn("test4")
    lan = Lang(target, model2.alphabet)
    # starting_examples = ["", "ab"]
    #
    # print(model2.is_word_in("ab"))
    # model2.is_word_in("aabababbababaaaaaaaaaaaaabbbbbbbbbabababababaa")
    #
    # lstar_dfa = extract(model2, time_limit=50, initial_split_depth=20, starting_examples=starting_examples)
    # teacher_pac = PACTeacher(model2)
    # student_pac = DecisionTreeLearner(teacher_pac)
    # teacher_pac.teach(student_pac)
    c = [True, True, True]
    d = [True, False, True]
    print([x == y for x in c for y in d])

    a, samples = confidence_interval_many([lan, model2], random_word)
    b, samples = confidence_interval(lan, model2, random_word, samples=samples)

    print("a")
    print(a)
    print("b")
    print(b)


print("blabla")
# learn_dfa_and_compare_distance("models2")
main_train_RNNS()
# main()
# cProfile.run('main()')
