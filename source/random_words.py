import sys

import numpy as np
import torch


def random_word(alphabet, p=0.01):
    nums_of_letters = len(alphabet)
    word = []
    while np.random.randint(0, int(1 / p)) != 0:
        letter = np.random.randint(0, nums_of_letters)
        word.append(alphabet[letter])
    return tuple(word)


def random_word_by_letter(alphabet, p=0.01):
    nums_of_letters = len(alphabet)
    while np.random.randint(0, int(1 / p)) != 0:
        letter = np.random.randint(0, nums_of_letters)
        yield alphabet[letter]


def confidence_interval(language1, language2, sampler, delta=0.001, epsilon=0.001, samples=None):
    n = np.log(2 / delta) / (2 * epsilon * epsilon)
    print(n)
    if samples is None:
        samples = set()
        while len(samples) < n:
            w = sampler(language1.alphabet)
            if w not in samples:
                samples.add(w)
            # print(len(samples))
    mistakes = 0
    print("got it")
    for w in samples:
        if language1.is_word_in(w) != language2.is_word_in(w):
            mistakes = mistakes + 1
            # print(mistakes)
    return mistakes / n, samples


def confidence_interval_many(languages, sampler, delta=0.001, epsilon=0.005, samples=None):
    num_of_lan = len(languages)
    if num_of_lan < 2:
        raise Exception("Need at least 2 languages to compare")

    n = np.log(2 / delta) / (2 * epsilon * epsilon)
    print("size of sample:" + str(int(n)))
    if samples is None:
        samples = set()
        while len(samples) <= n:
            if len(samples) % 1000 == 0:
                sys.stdout.write('\r Creating words:  {}/100 done'.format(str(int((len(samples) / n) * 100))))
            w = sampler(languages[0].alphabet)
            if w not in samples:
                if w != "":
                    samples.add(w)
        sys.stdout.write('\r Creating words:  100/100 done \n')
    in_langs_lists = []
    i = 0
    sys.stdout.write('\r Creating bool lists for each lan:  {}/{} done'.format(i, num_of_lan))
    torch.cuda.empty_cache()
    for lang in languages:
        # if isinstance(lang, LSTMLanguageClasifier):
        #     batch = []
        #     samplesL = list(samples)
        #     p = int(len(samples)/100)
        #     h = int(len(samples) / p)
        #     print(h)
        #     for j in range(p):
        #         print(j)
        #         b = lang.is_words_in_batch(samplesL[j * h: (j + 1) * h])
        #         batch.extend(b)
        #         torch.cuda.empty_cache()
        #     batch.extend(lang.is_words_in_batch(samplesL[p * h: len(samples)]))
        #     l = [bool(x > 0.5) for x in batch]
        #     in_langs_lists.append(l)
        #
        # else:
        in_langs_lists.append([lang.is_word_in(w) for w in samples])
        i = i + 1
        sys.stdout.write('\r Creating bool lists for each lan:  {}/{} done'.format(i, num_of_lan))

    output = []
    for i in range(num_of_lan):
        output.append([1] * num_of_lan)

    for lang1 in range(num_of_lan):
        for lang2 in range(num_of_lan):
            if lang1 == lang2:
                output[lang1][lang2] = 0
            elif output[lang1][lang2] == 1:
                output[lang1][lang2] = ([(in_langs_lists[lang1])[i] == (in_langs_lists[lang2])[i] for i in
                                         range(len(samples))].count(False)) / n

    print()
    return output, samples
