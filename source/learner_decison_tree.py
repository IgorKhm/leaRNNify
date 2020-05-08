import cProfile
import sys
import time

from dfa import DFA
from learner import Learner
from graphviz import Digraph

from pac_teacher import PACTeacher


class TreeNode:
    def __init__(self, name=tuple(), inLan=True, parent=None):
        self.right = None
        self.left = None
        self.name = name
        self.inLan = inLan
        self.parent = parent

        if parent is not None:
            self.depth = parent.depth + 1
        else:
            self.depth = 0

    def __repr__(self):
        return "TreeNode: name: \'" + self.name + "\', depth:" + str(self.depth)

    def draw(self, filename, max_depth=200):
        graph = Digraph('G', filename=filename)
        front = []

        if self.inLan:
            graph.node(self.name, color="blue")
        else:
            graph.node(self.name, color="red")
        if self.left is not None:
            front.append(self.left)
        if self.right is not None:
            front.append(self.right)

        while len(front) != 0:
            v = front.pop(0)
            is_leaf = True

            if v.left is not None:
                front.append(v.left)
                is_leaf = False
            if v.right is not None:
                front.append(v.right)
                is_leaf = False

            if is_leaf:
                name = v.name + "[leaf]"
            else:
                name = v.name

            if v.inLan:
                graph.node(name, color="blue")
            else:
                graph.node(name, color="red")

            if v.parent.left == v:
                graph.edge(v.parent.name, name, color="red", label="l")
            else:
                graph.edge(v.parent.name, name, color="blue", label="r")

        graph.view()


class DecisionTreeLearner(Learner):
    def __init__(self, teacher):
        self.teacher = teacher
        self._root = TreeNode(inLan=teacher.membership_query(tuple()))
        self._leafs = [self._root]
        self.dfa = self._produce_hypothesis()

    def _sift(self, word):
        current_node = self._root
        while True:
            if current_node in self._leafs:
                return current_node

            if self.teacher.membership_query(word + current_node.name):
                current_node = current_node.right
            else:
                current_node = current_node.left

    # def _sift_set(self, words: []):
    #     words_left = len(words)
    #     current_nodes = [self._root for _ in (range(words_left))]
    #     batch = [None for _ in words]
    #     while True:
    #         for i in range(len(words)):
    #             if words[i] is not None:
    #                 batch[i] = words[i] + current_nodes[i].name
    #         ans = self.teacher.model.is_words_in_batch(batch)
    #         j = 0
    #         for i in range(len(words)):
    #             if words[i] is not None:
    #                 if ans[j] > 0.5:
    #                     current_nodes[i] = current_nodes[i].right
    #                 else:
    #                     current_nodes[i] = current_nodes[i].left
    #                 j = j + 1
    #                 if current_nodes[i] in self._leafs:
    #                     words[i] = None
    #                     batch[i] = None
    #                     words_left = words_left - 1
    #         if words_left == 0:
    #             return current_nodes

    def _sift_set(self, words: []):
        words_left = len(words)
        current_nodes = [[self._root, i] for i in (range(words_left))]
        final = [None for _ in words]
        while True:
            anwsers = self.teacher.model.is_words_in_batch([words[x[1]] + x[0].name for x in current_nodes])

            for i in range(len(anwsers) - 1, -1, -1):
                if anwsers[i] > 0.5:
                    current_nodes[i][0] = current_nodes[i][0].right
                else:
                    current_nodes[i][0] = current_nodes[i][0].left

                if current_nodes[i][0] in self._leafs:
                    final[current_nodes[i][1]] = current_nodes[i][0]
                    del (current_nodes[i])
                    words_left = words_left - 1
            if words_left == 0:
                return final

    def _produce_hypothesis_set(self):
        transitions = {}
        final_nodes = []
        sl = []
        for leaf in self._leafs:
            if leaf.inLan:
                final_nodes.append(leaf.name)
            for l in self.teacher.alphabet:
                sl.append(leaf.name + l)
        states = self._sift_set(sl)
        for leaf in range(len(self._leafs)):
            tran = {}
            for l in range(len(self.teacher.alphabet)):
                tran.update({self.teacher.alphabet[l]: states[leaf * len(self.teacher.alphabet) + l].name})
            transitions.update({self._leafs[leaf].name: tran})

        return DFA("", final_nodes, transitions)

    def _produce_hypothesis(self):
        transitions = {}
        final_nodes = []
        for leaf in self._leafs:
            if leaf.inLan:
                final_nodes.append(leaf.name)
            tran = {}
            for l in self.teacher.alphabet:
                state = self._sift(leaf.name + tuple([l]))
                tran.update({l: state.name})
            transitions.update({leaf.name: tran})

        return DFA(tuple(""), tuple(final_nodes), transitions)

    def new_counterexample(self, w, set=False):
        first_time = False
        if len(self._leafs) == 1:
            new_differencing_string, new_state_string, first_time = self._leafs[0].name, w, True

        else:
            s = self.dfa.init_state
            prefix = tuple()
            for l in w:
                prefix = prefix + tuple([l])
                n = self._sift(prefix)
                s = self.dfa.next_state_by_letter(s, l)
                if n.name != s:
                    for n2 in self._leafs:
                        if n2.name == s:
                            break
                    n = finding_common_ancestor(n, n2)
                    new_differencing_string = tuple([l]) + n.name
                    break

            new_state_string = prefix[0:len(prefix) - 1]

        node_to_replace = self._sift(new_state_string)
        # try:
        if self.teacher.membership_query(node_to_replace.name + new_differencing_string):
            node_to_replace.left = TreeNode(new_state_string, first_time ^ node_to_replace.inLan, node_to_replace)
            node_to_replace.right = TreeNode(node_to_replace.name, node_to_replace.inLan, node_to_replace)
        else:
            node_to_replace.right = TreeNode(new_state_string, first_time ^ node_to_replace.inLan, node_to_replace)
            node_to_replace.left = TreeNode(node_to_replace.name, node_to_replace.inLan, node_to_replace)
        # except :
        #     print(sys.exc_info()[0])
        self._leafs.remove(node_to_replace)
        node_to_replace.name = new_differencing_string
        self._leafs.extend([node_to_replace.right, node_to_replace.left])

        # t = time.time()
        if set:
            self.dfa = self._produce_hypothesis_set()
        else:
            self.dfa = self._produce_hypothesis()

        # if self._produce_hypothesis_set() != self._produce_hypothesis():
        #     raise NotImplemented
        #     # print("check")

        # print(" time for prod hypothesis: {}".format(time.time() - t))
        # if self.dfa.is_word_in(prefix) != self._teacher.membership_query(prefix):
        #     print("?")


def finding_common_ancestor(node1: TreeNode, node2: TreeNode):
    if node1.depth < node2.depth:
        while node1.depth != node2.depth:
            node2 = node2.parent
    else:
        while node1.depth != node2.depth:
            node1 = node1.parent

    while node1 != node2:
        node1, node2 = node1.parent, node2.parent

    return node1
