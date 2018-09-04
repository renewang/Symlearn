# install cython
# encoding: utf-8
# cython: profile=True
# filename: cnltk.pyx
import logging
from collections import namedtuple
from nltk.compat import string_types

import re
import sys
import mmap
import os

logger = logging.getLogger(__name__)

logger.info('using cython fromstring for parsing')
Node = namedtuple('Node', ['label', 'children'])
data_dir = os.path.join(os.getenv('WORKSPACE'), 'Kaggle/symlearn/data/')

def fromstring(s, brackets='()', read_node=None, read_leaf=None,
              node_pattern=None, leaf_pattern=None,
              remove_empty_top_bracketing=False):
    """
    a cython rewrite for nltk.fromstring
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
    a cython rewrite for nltk._parse_error
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


def read_bin_trees(filename=None, selected_indices=None, converted=True,
        start_pos=0):
    """
    read tree as mmap file
    @param filename is a string referring to the memory mapped file location
    @param selected_indices is a iterable whose elements are the index of nltk
                            trees
    @param trees is a collection of trees which just for the purpose checking
    program
    """

    selected_trees = []
    if selected_indices is not None:
        line_no = 0

        def select_byindices(line, *args, **kwargs):
            nonlocal line_no
            is_selected = 0
            if line_no in selected_indices:
                is_selected = 1
                if converted:
                    selected_trees.append(fromstring(line))
                else:
                    selected_trees.append(line)
            if len(selected_trees) == len(selected_indices):
                is_selected = -1  # breaking loop
            line_no += 1
            return(is_selected)

        select_func = select_byindices
    else:
        select_func = None

    if filename is None:
        filename = os.path.join(data_dir, 'trees.bin')
    stop_pos = read_bin_file(filename, select_func, start_pos=start_pos)
    return(selected_trees)


def read_bin_file(filename, select_func=None, start_pos=0):
    """
    read file as mmap file
    """

    def select_all(line=None, *args, **kwargs):
        return(True)

    if select_func is None:
        select_func = select_all

    with open(filename, 'r+b') as f:
        mm = mmap.mmap(f.fileno(), 0)
        pos = start_pos 
        mm.seek(pos)
        mm.read()  # read in all files, try to figure out how large the file is
        size = mm.tell()
        mm.seek(pos)  # move back to the beginning
        line_no = 0
        while pos < size:
            line = mm.readline().decode(encoding='utf-8')
            sig = select_func(line)
            if sig < 0:
                break
            pos = mm.tell()  # read in one tree. report current position
            line_no += 1  # adding corresponding index
    return(pos)


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
