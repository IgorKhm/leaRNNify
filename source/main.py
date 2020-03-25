from copy import deepcopy

from tulip.transys import automata

from dfa import DFA, random_dfa
from exact_teacher import ExactTeacher
from learner_decison_tree import DecisionTreeLearner
from pac_teacher import PACTeacher

if __name__ == '__main__':
    pass

#############################################
#############################################
#############################################
#
#
#This is just a general nonsense area for me you 
# are welcome to ignore this file -ik
#
#
#############################################
#############################################
#############################################


dfa = DFA(1, {1}, {1: {"a": 2, "b": 1, "c": 1},
                   2: {"a": 3, "b": 1, "c": 3},
                   3: {"a": 3, "b": 3, "c": 3}})

dfa2 = DFA(1, {1, 4}, {1: {"a": 2, "b": 1, "c": 1},
                       2: {"a": 3, "b": 1, "c": 3},
                       3: {"a": 3, "b": 3, "c": 4},
                       4: {"a": 3, "b": 3, "c": 4}})

dfaKV = DFA(1, {4}, {1: {"0": 1, "1": 2},
                     2: {"0": 2, "1": 3},
                     3: {"0": 3, "1": 4},
                     4: {"0": 4, "1": 1}})

dfabug = DFA(0, {2, 3}, {0: {"0": 7, "1": 3},
                         1: {"0": 6, "1": 2},
                         2: {"0": 0, "1": 5},
                         3: {"0": 2, "1": 2},
                         4: {"0": 0, "1": 3},
                         5: {"0": 0, "1": 4},
                         6: {"0": 5, "1": 2},
                         7: {"0": 3, "1": 7}})


a= False
b = False
c = not a^b


# dfa.draw_nicely()
#
# b = random_word_by_letter(["a", "b", "c"])
#
# while True:
#     print(next(b))

#
# print(a[0:len(a) - 1])
# # b= DecisionTreeLearner()
a = PACTeacher(dfabug)


# # a.membership_query("abc")
#
# s = ""
# s2 = "a"
# print(s + s2)
#
student = DecisionTreeLearner(a)
# print("yoy")
#
# a.model.draw_nicely(name="1")

while True:
    # student.dfa.draw_nicely(name="2")
    counter = a.equivalence_query(student.dfa)
    if counter is None:
        break
    if student.dfa.is_word_in(counter) == a.model.is_word_in(counter) :
        print("?????")
    print(len(student.dfa.states))
    if len(student.dfa.states) == 6:
        print("")
    student.new_counterexample(counter)

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

#
while True:
    dfa3 = random_dfa(["a", "b", "c", "d"])
    dfa3.draw_nicely(name="1")

    teacher = ExactTeacher(dfa3)
    # a.membership_query("abc")

    student = DecisionTreeLearner(teacher)
    student.dfa.draw_nicely(name="2")
    counter = ""
    number_states = len(student.dfa.states)
    while True:
        counter2 = teacher.equivalence_query(student.dfa)
        if counter2 == counter:
            print("1?")
        counter = counter2
        if counter is None:
            break
        print(counter)
        student.new_counterexample(counter)
        if len(student.dfa.states) == number_states:
            print("not changing")
        number_states = len(student.dfa.states)
        # student.dfa.draw_nicely(name="2")

    student.dfa.draw_nicely(name="2")
    if dfa3 != student.dfa:
        print("not equal")
    if len(student.dfa.states) > 50:
        break
