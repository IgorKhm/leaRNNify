import cython
from libc.stdlib cimport rand, RAND_MAX

@cython.boundscheck(False)

def random_words(int num_of_words,tuple alphabet, double one_div_p=100.0):
    words = []
    cdef int nums_of_letters = len(alphabet)
    # cdef int one_div_p = int(1/p)
    print("rand()")
    print(rand()/(RAND_MAX/one_div_p))
    cdef int letter
    cdef int a
    for i in range(num_of_words):
        word = []
        # print(a)
        a = int(rand()/(RAND_MAX/one_div_p))
        while a != 0:
            a = int(rand()/(RAND_MAX/one_div_p))
            # letter = np.random.randint(0, nums_of_letters)
            letter =  int(rand()/(RAND_MAX/nums_of_letters))
            word.append(alphabet[letter])

        words.append(tuple(word))
    return words
