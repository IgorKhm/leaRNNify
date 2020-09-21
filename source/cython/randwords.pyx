import cython
from libc.stdlib cimport rand, RAND_MAX

@cython.boundscheck(False)
#
def random_words(int num_of_words,tuple alphabet, double one_div_p=100.0):
    words = []
    cdef int nums_of_letters = len(alphabet)
    # cdef int one_div_p = int(1/p)
    print("rand()")
    print((RAND_MAX/one_div_p))
    cdef int letter
    cdef int a
    for i in range(num_of_words):
        word = []
        # print(a)
        a = int(rand()/(RAND_MAX/one_div_p))
        while a != 0:
            a = int(rand()/(RAND_MAX/one_div_p))
            # letter = np.random.randint(0, nums_of_letters)
            letter =  int(rand()/(RAND_MAX/nums_of_letters +1))
            # if letter >= nums_of_letters:
            #   print("problem")
            #   print(letter)
            #   print(len(alphabet))
            #   print(nums_of_letters)
            word.append(alphabet[letter])

        words.append(tuple(word))
    return words

def is_words_in_dfa(dfa, words):
    lenwords = len(words)
    outputs = []
    final_states = dfa.final_states
    init_state = dfa.init_state
    transitions = dfa.transitions
    for i in range(lenwords):
        state = init_state
        word = words[i]
        lenword= len(word)
        for j in range(lenword):
            state = transitions[state][word[j]]

        outputs.append(state in final_states)

    return outputs

def compare_list_of_bool(lang1, lang2,num_of_samples):
        # lenwords = len(lang1)
        cdef int count = 0
        for i in range(num_of_samples):
            count += (lang1[i]!=lang2[i])


        return (count/num_of_samples)


    # ([(in_langs_lists[lang1])[i] == (in_langs_lists[lang2])[i] for i in
    #                                         range(len(samples))].count(False)) / num_of_samples    # state = self.init_state
    # #         for letter in word:
    #             state = self.transitions[state][letter]
    #         return state in self.final_states
