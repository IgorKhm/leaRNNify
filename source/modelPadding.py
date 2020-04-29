import os

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
from torch.utils.data import Dataset


class WordsDataset(Dataset):
    def __init__(self, words, labels):
        self.labels = labels
        self.words = words

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.from_numpy(self.words[idx]), self.labels[idx]


def teach(model, batch_size, train_loader, val_loader, device, lr=0.005, criterion=nn.BCELoss(),
          epochs=10, print_every=500):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    print(model)
    print(optimizer)
    print(lr)
    counter = 0
    clip = 5
    valid_loss_min = np.Inf

    # val_h = model.init_hidden(batch_size)
    # val_losses = []
    # model.eval()
    # for inp, lab , inp_len, lab_len in val_loader:
    #     val_h = tuple([each.data for each in val_h])
    #     inp, lab = inp.to(device), lab.to(device)
    #     out, _ = model(inp, val_h)
    #     val_loss = criterion(out.squeeze(), lab.float())
    #     val_losses.append(val_loss.item())
    #
    model.train()
    # epochval = np.mean(val_losses)
    # print("Initial Val Loss: {:.6f}".format(np.mean(val_losses)))

    for i in range(epochs):
        h = model.init_hidden(batch_size)

        for inputs, labels, inp_len in train_loader:
            counter += 1
            h = tuple([e.data for e in h])

            inputs, labels = inputs.to(device), labels.to(device)
            model.zero_grad()
            output, _ = model(inputs, inp_len, h)
            loss = criterion(output.squeeze(), labels.float())
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()

            if counter % print_every == 0:
                val_h = model.init_hidden(batch_size)
                val_losses = []
                model.eval()
                num_correct = 0
                for inp, lab, inp_len in val_loader:
                    val_h = tuple([each.data for each in val_h])
                    inp, lab = inp.to(device), lab.to(device)
                    out, _ = model(inp, inp_len, val_h)
                    val_loss = criterion(out.squeeze(), lab.float())
                    val_losses.append(val_loss.item())

                    pred = torch.round(output.squeeze())  # rounds the output to 0/1
                    correct_tensor = pred.eq(labels.float().view_as(pred))
                    correct = np.squeeze(correct_tensor.cpu().numpy())
                    num_correct += np.sum(correct)

                test_acc = num_correct / len(val_loader.dataset)
                model.train()
                print("Epoch: {}/{}...".format(i + 1, epochs),
                      "Step: {}...".format(counter),
                      "Loss: {:.6f}...".format(loss.item()),
                      "Val Loss: {:.6f}".format(np.mean(val_losses)))
                print("Test accuracy: {:.3f}%".format(test_acc * 100))
                if np.mean(val_losses) <= valid_loss_min:
                    torch.save(model.state_dict(), './state_dict.pt')
                    print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min,
                                                                                                    np.mean(
                                                                                                        val_losses)))
                    valid_loss_min = np.mean(val_losses)
                    if valid_loss_min < 0.001:
                        return model
    return model


def test_rnn(model, test_loader, batch_size, device, criterion=nn.BCELoss()):
    # Loading the best model
    model.load_state_dict(torch.load('./state_dict.pt'))

    test_losses = []
    num_correct = 0
    h = model.init_hidden(batch_size)

    model.eval()
    for inputs, labels, len_inp in test_loader:
        h = tuple([each.data for each in h])
        inputs, labels = inputs.to(device), labels.to(device)
        output, _ = model(inputs, len_inp, h)
        test_loss = criterion(output.squeeze(), labels.float())
        test_losses.append(test_loss.item())
        pred = torch.round(output.squeeze())  # rounds the output to 0/1
        correct_tensor = pred.eq(labels.float().view_as(pred))
        correct = np.squeeze(correct_tensor.cpu().numpy())
        num_correct += np.sum(correct)

    print("Test loss: {:.3f}".format(np.mean(test_losses)))
    test_acc = num_correct / len(test_loader.dataset)
    print("Test accuracy: {:.3f}%".format(test_acc * 100))


def from_array_to_word(int2char, array):
    word = ""
    for i in array:
        word = word + int2char[i]
    return word


def pad_collate(batch):
    (xx, yy) = (zip(*batch))

    x_lens = [len(x) for x in xx]
    # y_lens = [len(y) for y in yy]

    xx_pad = pad_sequence(xx, batch_first=True, padding_value=0)
    # yy_pad = pad_sequence(yy, batch_first=True, padding_value=0)

    return xx_pad, torch.tensor(yy), x_lens  # , y_lens


def make_training_sets(alphabet, target, num_of_exm_per_length=2000, max_length=50,
                       batch_size=50):
    int2char = ({i + 1: alphabet[i] for i in range(len(alphabet))})
    int2char.update({0: ""})
    char2int = {alphabet[i]: i + 1 for i in range(len(alphabet))}
    char2int.update({"": 0})
    words_list = []
    lengths = list(range(1, 20)) + list(range(20, max_length, 5))
    for length in lengths:
        new_list = np.unique(np.random.randint(1, len(alphabet) + 1, size=(num_of_exm_per_length, length)), axis=0)
        words_list.extend(new_list)

    label_list = [target(from_array_to_word(int2char, w)) for w in words_list]

    print(len(words_list))
    test_label, test_words, train_label, train_words, val_label, val_words = \
        _split_words_to_train_val_and_test(batch_size, label_list, words_list)

    train_data = WordsDataset(train_words, train_label)
    val_data = WordsDataset(val_words, val_label)
    test_data = WordsDataset(test_words, test_label)

    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size, collate_fn=pad_collate)
    val_loader = DataLoader(val_data, shuffle=True, batch_size=batch_size, collate_fn=pad_collate)
    test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size, collate_fn=pad_collate)

    return train_loader, val_loader, test_loader


def _split_words_to_train_val_and_test(batch_size, label_list, words_list):
    # chooses 10% words for test examples, and we want it to devied the bantch size
    num_test = int(len(words_list) / 10)
    num_test = num_test - num_test % batch_size
    test_words, test_label = [], []
    for _ in range(num_test):
        i = np.random.randint(0, len(words_list))
        test_words.append(words_list[i])
        test_label.append(label_list[i])
        # del words_list[i]
        # del label_list[i]

    # split the rest between validation and learning
    num_val = int(len(words_list) / 4)
    num_val = num_val - num_val % batch_size
    val_words, val_label = [], []
    for _ in range(num_val):
        i = np.random.randint(0, len(words_list))
        val_words.append(words_list[i])
        val_label.append(label_list[i])
        # del words_list[i]
        # del label_list[i]
    print(len(words_list) % batch_size)
    train_num = int(len(words_list) - len(words_list) % batch_size)
    train_words, train_label = words_list[0:train_num], label_list[0:train_num]
    return test_label, test_words, train_label, train_words, val_label, val_words


class LSTM(nn.Module):
    def __init__(self, alphabet_size, output_size, embedding_dim, hidden_dim, n_layers, drop_prob=0.5,
                 device=torch.device("cpu")):
        super(LSTM, self).__init__()
        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim

        self.embedding = nn.Embedding(alphabet_size, embedding_dim).to(device=device)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=drop_prob, batch_first=True).to(device=device)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_dim, output_size).to(device=device)
        self.sigmoid = nn.Sigmoid().to(device=device)
        self.device = device

    def forward(self, x, x_lens, hidden):
        # with torch.no_grad():
        # self.eval()
        batch_size = x.size(0)
        x = x.long()
        # x_paded = pad_sequence(x, batch_first=True, padding_value=0)
        x_embed = self.embedding(x)

        x_packed = pack_padded_sequence(x_embed, x_lens, batch_first=True, enforce_sorted=False)
        output_padded, hidden = self.lstm(x_packed, hidden)

        out_ltsm, output_lengths = pad_packed_sequence(output_padded, batch_first=True)
        out_ltsm = out_ltsm.contiguous().view(-1, self.hidden_dim)

        out = self.dropout(out_ltsm)
        out = self.fc(out)
        out = self.sigmoid(out)

        out = out.view(batch_size, -1)
        out = out[:, -1]
        return out, hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(self.device),
                  weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(self.device))
        return hidden


class LSTMLanguageClasifier:
    def __init__(self):
        self._ltsm = None
        self._initial_state = None
        self._current_state = None
        self._char_to_int = None
        self.alphabet = []
        self.word_traning_length = 40

    def train_a_lstm(self, alphahbet, target, embedding_dim=10, hidden_dim=10, num_layers=2, batch_size=20,
                     num_of_exm_per_lenght=5000, epoch=20):
        self._char_to_int = {alphahbet[i]: i + 1 for i in range(len(alphahbet))}
        self._char_to_int.update({"": 0})
        self.alphabet = alphahbet

        is_cuda = torch.cuda.is_available()
        if is_cuda:
            device = torch.device("cuda")
            print("GPU is available")
        else:
            device = torch.device("cpu")
            print("GPU not available, CPU used")

        self._ltsm = LSTM(len(alphahbet) + 1, 1, embedding_dim, hidden_dim, num_layers, drop_prob=0.5,
                          device=device)
        train_loader, val_loader, test_loader = make_training_sets(alphahbet, target, batch_size=batch_size,
                                                                   num_of_exm_per_length=num_of_exm_per_lenght,
                                                                   max_length=self.word_traning_length)
        print(len(train_loader))
        try:
            self._ltsm = teach(self._ltsm, batch_size, train_loader, val_loader, device, epochs=epoch, print_every=2000)
        except KeyboardInterrupt():
            print("Training of the RNN was stopped by user. Continuing with the rest")
        self._initial_state = self._ltsm.init_hidden(1)
        self._current_state = self._initial_state

        test_rnn(self._ltsm, test_loader, batch_size, device)
        return test_loader

    def is_word_in(self, word):
        h = self._ltsm.init_hidden(1)
        if len(word) < self.word_traning_length:
            length = self.word_traning_length
        else:
            length = len(word)
        array = np.zeros(length)
        for i in range(len(word)):
            array[length - i - 1] = self._char_to_int[word[-i - 1]]
        array = np.array([array])
        if len(word) == 0:
            l = torch.from_numpy(np.array([[0]]))
            output, h = self._ltsm(l.to(self._ltsm.device), h)
        else:
            output, h = self._ltsm(torch.from_numpy(array).to(self._ltsm.device), h)
        return bool(output > 0.5)

    def is_word_in_test(self, word, lengthadditon):
        h = self._ltsm.init_hidden(1)
        if len(word) < self.word_traning_length:
            length = self.word_traning_length + lengthadditon
        else:
            length = len(word) + lengthadditon
        array = np.zeros(length)
        for i in range(len(word)):
            array[length - i - 1] = self._char_to_int[word[-i - 1]]
        array = np.array([array])
        if len(word) == 0:
            l = torch.from_numpy(np.array([[0]]))
            output, h = self._ltsm(l.to(self._ltsm.device), h)
        else:
            output, h = self._ltsm(torch.from_numpy(array).to(self._ltsm.device), h)
        return bool(output > 0.5)

    # def is_words_in_batch_test(self, words):
    #     h = self._ltsm.init_hidden(len(words))
    #     words.remove("")
    #     batches = []
    #     max_len = len(max(words, key=lambda w: len(w)))
    #     for word in words:
    #         if max_len < 100:
    #             length = 100
    #         else:
    #             length = max_len
    #         array = np.zeros(length)
    #         for i in range(len(word)):
    #             array[length - i - 1] = self._char_to_int[word[-i - 1]]
    #         batches.append(array)
    #     batch_np = np.array(batches)
    #     batch = torch.from_numpy(batch_np)
    #     batch = words_list = torch.nn.utils.rnn.pack_padded_sequence(batch, [len(w) for w in words],enforce_sorted=False,batch_first=True)
    #     # if len(word) == 0:
    #     #     l = torch.from_numpy(np.array([[0]]))
    #     #     output, h = self._ltsm(l.to(self._ltsm.device), h)
    #     # else:
    #     output, _ = self._ltsm(batch.to(self._ltsm.device), h)
    #     return output
    #
    def is_words_in_batch(self, words):
        batches = []
        # for i in range(len(words)/10000):
        #     batches.append(len(words)/10000)
        #
        #
        #
        max_len = len(max(words, key=lambda w: len(w)))
        for word in words:
            if max_len < self.word_traning_length:
                length = self.word_traning_length
            else:
                length = max_len
            array = np.zeros(length)
            for i in range(len(word)):
                array[length - i - 1] = self._char_to_int[word[-i - 1]]
            batches.append(array)
        batch_np = np.array(batches)
        # if len(word) == 0:
        #     l = torch.from_numpy(np.array([[0]]))
        #     output, h = self._ltsm(l.to(self._ltsm.device), h)
        # else:
        h = self._ltsm.init_hidden(len(batch_np))
        output, _ = self._ltsm(torch.from_numpy(batch_np).to(self._ltsm.device), h)
        return output
        #
        # h = self._ltsm.init_hidden(1)
        # if len(word) == 0:
        #     l = torch.from_numpy(np.array([[0]]))
        #     output, h = self._ltsm(l, h)
        # else:
        #     for l in word:
        #         l = torch.from_numpy(np.array([[self._char_to_int[l]]]))
        #         output, h = self._ltsm(l, h)
        # return bool(output > 0.5)

    def is_word_letter_by_letter(self, letter):
        letter = torch.from_numpy(np.array([[self._char_to_int[letter]]]))
        output, self._current_state = self._ltsm(letter, self._current_state)
        return bool(output > 0.5)

    def reset_current_to_init(self):
        self._current_state = self._ltsm.init_hidden(1)

    def save_rnn(self, dirName, force_overwrite=False):
        if not os.path.isdir(dirName):
            os.makedirs(dirName)
        elif os.path.exists(dirName + "/meta") & (not force_overwrite):
            if input("save exists. Enter y if you want to overwrite it.") != "y":
                return
        with open(dirName + "/meta", "w+") as file:
            file.write("Metadata:\n")
            file.write("alphabet = " + self.alphabet + "\n")
            file.write("embedding_dim = " + str(self._ltsm.embedding_dim) + "\n")
            file.write("hidden_dim = " + str(self._ltsm.hidden_dim) + "\n")
            file.write("n_layers = " + str(self._ltsm.n_layers) + "\n")
            file.write("torch_save = state_dict.pt")
        torch.save(self._ltsm.state_dict(), dirName + "/state_dict.pt")

    def load_rnn(self, dir):
        # './state_dict.pt'
        is_cuda = torch.cuda.is_available()
        if is_cuda:
            device = torch.device("cuda")
            print("GPU is available")
        else:
            device = torch.device("cpu")
            print("GPU not available, CPU used")

        with open(dir + "/meta", "r") as file:
            # lines = file.
            for line in file.readlines():
                splitline = line.split(" = ")
                if splitline[0] == "alphabet":
                    self.alphabet = splitline[1].rstrip('\n')
                elif splitline[0] == "embedding_dim":
                    embedding_dim = int(splitline[1])
                elif splitline[0] == "hidden_dim":
                    hidden_dim = int(splitline[1])
                elif splitline[0] == "n_layers":
                    n_layers = int(splitline[1])
                elif splitline[0] == "torch_save":
                    torch_save = splitline[1].rstrip('\n')

        self._ltsm = LSTM(len(self.alphabet) + 1, 1, embedding_dim, hidden_dim, n_layers, drop_prob=0.5, device=device)
        self._ltsm.load_state_dict(torch.load(dir + "/" + torch_save))
        self._ltsm.eval()
        # self._ltsm.fc.eval()
        # torch.no_grad()

        self._initial_state = self._ltsm.init_hidden(1)
        self._current_state = self._initial_state
        self._char_to_int = {self.alphabet[i]: i + 1 for i in range(len(self.alphabet))}
        self._char_to_int.update({"": 0})
        self.states = {
            str(self.from_state_to_list(self._ltsm.init_hidden(1))): ""}  # maybe move to load? or some other place?

    ######################################################
    #                 Code For Lstar                     #
    ######################################################

    def classify_word(self, word):
        return bool(self.is_word_in(word))

    def get_first_RState(self):
        return self.from_state_to_list(self._ltsm.init_hidden(1)), bool(self.is_word_in(""))

    def get_next_RState(self, state, char):
        # state = self.from_list_to_state(state)
        # print(len(self.states.keys()))
        # print(self.states.values())
        word = self.states[str(state)] + char
        if len(word) < self.word_traning_length:
            length = self.word_traning_length
        else:
            length = len(word)
        array = np.zeros(length)
        for i in range(len(word)):
            array[length - i - 1] = self._char_to_int[word[-i - 1]]
        array = np.array([array])
        state = self._ltsm.init_hidden(1)
        if len(word) == 0:
            l = torch.from_numpy(np.array([[0]]))
            output, state = self._ltsm(l, state)
        else:
            output, state = self._ltsm(torch.from_numpy(array).to(device=self._ltsm.device), state)
            self.states.update({str(self.from_state_to_list(state)): word})

        # print(word)
        return self.from_state_to_list(state), bool(output > 0.5)

    def from_state_to_list(self, state):
        # hidden_dim = self._ltsm.hidden_dim
        list_state = []
        for i in state[0][0, 0]:
            list_state.append(float(i))
        for i in state[1][0, 0]:
            list_state.append(float(i))

        return list_state

    def from_list_to_state(self, list_state):

        hiden = torch.tensor([[list_state[self._ltsm.hidden_dim:]]])
        cell = torch.tensor([[list_state[:self._ltsm.hidden_dim]]])
        return (hiden, cell)

        # c = self._char_to_int[char]
        # c = torch.tensor([[c]])
        # output, state = self._ltsm(c, state)
        # return state, output > 0.5
