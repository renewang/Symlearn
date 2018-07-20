from collections import namedtuple
from operator import itemgetter
from nltk.compat import string_types
import re
import sys
import logging
import numpy
import runpy
import os
import networkx

logger = logging.getLogger(__name__)
try:
    import pyximport
    pyximport.install(reload_support=True, build_in_temp=False, inplace=True,
            language_level=3)
    from recursnn.cnltk import (fromstring, Node, encode_path, read_bin_trees, 
                                read_bin_file)
    logger.info('using cython module')
except ImportError:
    logger.info('using python module')
    Node = namedtuple('Node', ['label', 'children'])

    def fromstring(s, brackets='()', read_node=None, read_leaf=None,
              node_pattern=None, leaf_pattern=None,
              remove_empty_top_bracketing=False):
        """
        a modification from class method of nltk.Tree.fromstring to remove class 
        binding

        Parameters
        ==========
        see nltk.Tree.fromstring
        """
        if not isinstance(brackets, string_types) or len(brackets) != 2:
            raise TypeError('brackets must be a length-2 string')
        if re.search('\s', brackets):
            raise TypeError('whitespace brackets not allowed')

        # Construct a regexp that will tokenize the string.
        open_b, close_b = brackets
        open_pattern, close_pattern = (re.escape(open_b), re.escape(close_b))
        if node_pattern is None:
            node_pattern = '[^\s%s%s]+' % (open_pattern, close_pattern)
        if leaf_pattern is None:
            leaf_pattern = '[^\s%s%s]+' % (open_pattern, close_pattern)
        token_re = re.compile('%s\s*(%s)?|%s|(%s)' % (
            open_pattern, node_pattern, close_pattern, leaf_pattern))
        # Walk through each token, updating a stack of trees.
        stack = [(None, [])]  # list of (node, children) tuples
        for match in token_re.finditer(s):
            token = match.group()
            # Beginning of a tree/subtree
            if token[0] == open_b:
                if len(stack) == 1 and len(stack[0][1]) > 0:
                    _parse_error(s, match, 'end-of-string')
                label = token[1:].lstrip()
                if read_node is not None:
                    label = read_node(label)
                stack.append((label, []))
            # End of a tree/subtree
            elif token == close_b:
                if len(stack) == 1:
                    if len(stack[0][1]) == 0:
                        _parse_error(s, match, open_b)
                    else:
                        _parse_error(s, match, 'end-of-string')
                label, children = stack.pop()
                stack[-1][1].append(Node(label, children))
            # Leaf node
            else:
                if len(stack) == 1:
                    _parse_error(s, match, open_b)
                if read_leaf is not None:
                    token = read_leaf(token)
                stack[-1][1].append(token)

        # check that we got exactly one complete tree.
        if len(stack) > 1:
            _parse_error(s, 'end-of-string', close_b)
        elif len(stack[0][1]) == 0:
            _parse_error(s, 'end-of-string', open_b)
        else:
            assert stack[0][0] is None
            assert len(stack[0][1]) == 1
        tree = stack[0][1][0]

        # If the tree has an extra level with node='', then get rid of
        # it.  E.g.: "((S (NP ...) (VP ...)))"
        if remove_empty_top_bracketing and tree._label == '' and \
                len(tree) == 1:
            tree = tree[0]
        # return the tree.
        return tree

    def _parse_error(s, match, expecting):
        """
        a rewrite of nltk.tree.Tree method by remove its class binding

        Parameters
        ==========
        see nltk.Tree.fromstring
        """
        # Construct a basic error message
        if match == 'end-of-string':
            pos, token = len(s), 'end-of-string'
        else:
            pos, token = match.start(), match.group()
        msg = '%s.read(): expected %r but got %r\n%sat index %d.' % (
            __name__, expecting, token, ' ' * 12, pos)
        # Add a display showing the error token itsels:
        s = s.replace('\n', ' ').replace('\t', ' ')
        offset = pos
        if len(s) > pos + 10:
            s = s[:pos + 10] + '...'
        if pos > 10:
            s = '...' + s[pos - 10:]
            offset = 13
        msg += '\n%s"%s"\n%s^' % (' ' * 16, s, ' ' * (17 + offset))
        raise ValueError(msg)

    def encode_path(index, path):
        """
        encode the index keys into nltk.tree.Tree tree positions key

        Parameters
        ==========
        @params index is an non-negative integer and used to compute path
        @params path is a initial root path to construct finaly path result
        recursively

        >>> encode_path(8, ()) == (0, 0, 1)
        True
        >>> encode_path(9, ()) == (0, 1, 0)
        True
        >>> encode_path(16, ()) == (0, 0, 0, 1)
        True
        """
        count, max_blen = 0, index.bit_length()
        while index != 0 and count <= max_blen:
            path = ((index - 1) % 2,) + path
            index = (index - 1) // 2
            count += 1
        assert(index == 0)
        return(path)


def treepositions(self, order='preorder'):
    """
    a delegating function for nltk.tree.Tree.treepositions method

    Parameters
    ==========
    see nltk.Tree.fromstring
    """
    assert(isinstance(self, networkx.Graph))
    if order != 'preorder':
        raise NotImplementedError('{:s} is not implemented'.format(order))
    return(self.nodes)


def leaves(self):
    """
    a delegating function for nltk.tree.Tree.leaves method

    Parameters
    ==========
    see nltk.tree.Tree.leaves
    """
    # not really working since the node in networkx.Graph only in dict
    assert(isinstance(self, networkx.Graph))
    return([k for k, v in self.out_degree(self.nodes()).items() if v == 0])


def _check_allocation(shape, constructor, dtype=numpy.int32):
    """
    given shape checking if the size exceeding the allowed capacity

    Parameters
    ==========
    @param shape
    @param constructor
    @param dtype
    """
    try:
        mat = constructor(shape, dtype=dtype)
    except:
        raise
    return(mat)


def shift_to_zeros(indexer):
    """
    to get the level from the binary key as taking ceil on log2 value of
    bin_key + 1

    Parameters
    ==========
    @param indexer 
    """
    nodes = numpy.empty(len(indexer))
    vals, locs = list(zip(*indexer.items()))
    nodes[locs, ] = vals
    return(numpy.floor(numpy.log2(nodes + 1)).astype(numpy.int))


def get_levels_and_labels(graph, indexer):
    """

    Parameters
    ==========
    @param graph
    @param indexer
    """
    levels = shift_to_zeros(indexer)
    labels = numpy.empty_like(levels)
    lab_iter = numpy.nditer(labels, op_flags=['writeonly'], flags=['c_index'])
    max_num = 0
    while not lab_iter.finished:
        lab_iter[0] = graph[lab_iter.index]['_label']
        max_num += 1
        if max_num > len(labels):
            break
        lab_iter.iternext()
    return(levels, labels)


def compute_stat_result(results, max_level, n_classes=5):
    """
    compute tree statistics result

    Parameters
    ==========
    @param results, the pre-computed tree statistics result 
    @param max_level, int, the maximal level a parsing tree could have 
    @param n_classes, int, the total number of sentiment classes 
    """
    stat_result = numpy.zeros((max_level, n_classes, n_classes))
    tree_result = []
    for i in range(len(results)):
        graph, indexer = results[i]
        levels, labels = get_levels_and_labels(graph, indexer)
        _compute_stat_result(levels, labels, stat_result)
        tree_result.append((graph, indexer, levels, labels))
    return(stat_result, tree_result)


def _compute_stat_result(levels, labels, stat_result):
    """
    compute tree statistics result

    Parameters
    ==========
    @param levels, numpy integer array
    @param labels, numpy integer array
    @param stat_result, numpy integer multi-dimensional array whose
    dimensionality is # of levels x # of root sentiment labels, x of node
    sentiment labels
    """
    roots = labels[0]
    # TOOD: inefficient loop
    for l, s in numpy.nditer([levels[levels > 0], labels[levels > 0]],
                            op_dtypes=[numpy.int, numpy.int],
                            flags=['buffered'],
                            casting='unsafe'):
        stat_result[l - 1, roots, s] += 1


def postprocess_results(results):
    """
    post-processing to extract levels and labels from statistics result

    Parameters
    ==========
    @param result
    """
    max_level = numpy.max([g.number_of_nodes()
        for g in map(itemgetter(0), results)])
    levels = numpy.zeros((len(results), max_level), dtype=numpy.int)
    labels = numpy.zeros((len(results), max_level), dtype=numpy.int)
    for i, (lev, labs) in enumerate(map(itemgetter(-2, -1), results)):
        levels[i][:len(lev)] = lev
        labels[i][:len(labs)] = labs
    return(levels, labels)


def _translate(idx):
    """
    translate integer into bit string

    Parameters
    ==========
    @param idx, integer, used to convert into bit string 
    """
    val = idx.to_bytes(numpy.dtype(numpy.int64).itemsize,
            byteorder=sys.byteorder)
    return(val)


def _inverse_translate(byte_key):
    """
    inversely translate bit string into integer 

    Parameters
    ==========
    @param byte_key, bit string used to convert into integer
    """
    val = int.from_bytes(byte_key, byteorder=sys.byteorder)
    return(val)

if __name__ == '__main__':
    import cProfile
    import pstats

    test_strs = ['(2 (0 tree) (1 (0 number) (0 one)))',
            '(2 (1 (0 tree) (0 number)) (0 two))',
            '(2 (1 (0 tree) (0 number)) (1 (0 three) (0 .)))',
            '(3 (2 (1 (0 tree) (0 number)) (0 four)) (0 .))',
            '(3 (0 tree) (2 (1 (0 number) (0 five)) (0 .)))']
    cProfile.run('tested_tree = [fromstring(s) for s in test_strs]',
            'cnltk.prof')
    prof_stat = pstats.Stats('cnltk.prof')
    prof_stat.strip_dirs().sort_stats(-1).print_stats()
