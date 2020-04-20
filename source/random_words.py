import string

import numpy as np
from IPython.display import Image
from IPython.display import display

import graphviz as gv
import functools

from numpy.random._generator import default_rng


def random_word(alphabet, p=0.01):
    nums_of_letters = len(alphabet)
    word = ""
    while np.random.randint(0, int(1 / p)) != 0:
        letter = np.random.randint(0, nums_of_letters)
        word = word + alphabet[letter]
    return word


def random_word_by_letter(alphabet, p=0.01):
    nums_of_letters = len(alphabet)
    while np.random.randint(0, int(1 / p)) != 0:
        letter = np.random.randint(0, nums_of_letters)
        yield alphabet[letter]


def confidence_interval(language1, language2, sampler, delta=0.001, epsilon=0.005, samples=None):
    n = np.log(2 / delta) / (2 * epsilon * epsilon)
    print(n)
    if samples is None:
        samples = set()
        while len(samples) < n:
            w = sampler(language1.alphabet)
            if w not in samples:
                samples.add(w)
            print(len(samples))
    mistakes = 0
    print("got it")
    for w in samples:
        if language1.is_word_in(w) != language2.is_word_in(w):
            mistakes = mistakes + 1
            # print(mistakes)
    return mistakes / n, samples
