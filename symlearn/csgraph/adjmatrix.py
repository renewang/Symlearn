from collections.abc import MutableMapping, Sequence 
from collections import OrderedDict, UserDict, deque, Counter
from operator import itemgetter
from itertools import repeat
from functools import cmp_to_key
from enum import IntEnum
from scipy import sparse

from symlearn.utils import get_phrases_helper
from nltk.tree import Tree

from . import cnltk_trees as cnltk

import array as parray
import networkx
import scipy
import numpy
import abc
import logging


logger = logging.getLogger(__name__)
Indexer = IntEnum('Indexer', [('tree', 0), ('matrix', 1)], module=__name__)

def enc_path_comp(p1, p2):
    if len(p1[1]) == len(p2[1]):
        return(p1[-1] - p2[-1])
    else:
        return(len(p2[1]) - len(p1[1]))


class Delegator(object):
    """
    a wrapper class to provide the delegated interface for delegating instance
    """

    def __init__(self, instances):
        """
        @param instances is an OrderedDict whose values are the delegating
        instances and whose keys are the identifier for delegating instances
        """
        assert(isinstance(instances, OrderedDict))
        self.delegators = instances
    
    def get_delegator(self, name):
        delegator = None
        # TODO: currently force to skip self.__class__ for isinstance checking
        # when passing into scipy.sparse.issparse and might look for re-arrange
        # mro to return True
        if hasattr(self, name) and name != '__class__':
            return(getattr(self, name))
        if name in self.delegators:
            return(self.delegators[name])
        else:
            for delegator in self.delegators.values():
                if hasattr(delegator, name):
                    return(getattr(delegator, name))
        if delegator is None:
            raise AttributeError('{!r} has no {:s}'.format(name))

    def visit_inodes_inorder(self, root=0):
        """
        given a tree matrix and return non-terminal nodes in indexing order
        the root will be returned at the last elements disregarding any
        indexing scheme

        @param matrix is the tree matrix in scipy.sparse format
        """
        target = self.delegators['csgraph']
        # the index of non-terminals, the word order will be messed up 
        nt_locs = numpy.arange(target.shape[0])[target.diagonal() == 1]
        # getting index of non-terminals 
        tree_indexer = numpy.asarray(
                self.delegators['indexer'].index_pairs[Indexer.tree])
        tree_indx = list(map(int, tree_indexer[nt_locs]))
        enc_path = map(cnltk.encode_path, tree_indx , repeat(()))
        resorted = sorted([(loc, path, idx) for loc, path, idx in
            zip(nt_locs, enc_path, tree_indx)], key=cmp_to_key(enc_path_comp))
        locs, _, _ = zip(*resorted)
        return(numpy.asarray(locs))

    def visit_leaves_inorder(self):
        """
        recover leaves and re-arange them in the order within the sentence
        """
        target = self.delegators['csgraph']
        # getting locations of leaves
        leaf_locs = numpy.arange(target.shape[0])[target.diagonal() > 1]
        # getting index of leaves
        tree_indexer = numpy.asarray(
                self.delegators['indexer'].index_pairs[Indexer.tree])
        enc_path = map(cnltk.encode_path,
                       map(int, tree_indexer[leaf_locs]), repeat(()))
        resorted = sorted([(loc, path) for loc, path in
                          zip(leaf_locs, enc_path)], key=itemgetter(1))
        return(resorted)

    def leaves(self):
        """
        an emulating function for nltk.tree.Tree.leaves method

        @ returns a list of indices to retrieve word in vocab provided by
        client
        """
        locs, _ = zip(*self.visit_leaves_inorder())
        vocab_inds = self.delegators['csgraph'].diagonal()[locs, ] - 2
        assert(numpy.all(vocab_inds >= 0))
        assert(len(vocab_inds) == len(set(locs)))
        return(vocab_inds)

    def treepositions(self, order='preorder'):
        """
        an emulating function for nltk.tree.Tree.treepositions method

        @ returns a list of indices can map to nltk.tree.Tree._index attribute
        """
        matrix_indexer = numpy.asarray(self.delegators[
            'indexer'].index_pairs[Indexer.matrix])
        tree_indexer = numpy.asarray(self.delegators[
            'indexer'].index_pairs[Indexer.tree])
        if order == 'preorder':
            return(list(map(bin, tree_indexer[matrix_indexer])))
        elif order == 'leaf':
            locs, _ = zip(*self.visit_leaves_inorder())
            return(list(map(bin, tree_indexer[locs])))
        else:
            raise NotImplementedError('%s is not yet implemented' % order)

    def calculate_spanweight(self, all_spans=None):
        """
        calculate the word span for each node. For terminal node the span is zero
        """
        if all_spans is None:
            all_spans = self.get_spanrange()

        span_weight = numpy.ones(len(all_spans), dtype=numpy.int)
        span_mask = span_weight.view(numpy.ma.MaskedArray)
        span_mask.mask = all_spans.mask.all(axis=1)
        nt_inds = numpy.arange(len(span_weight))[~span_mask.mask]
        for i, span_range in enumerate(all_spans.compressed().reshape(-1, 2)):
            # avoiding zero
            span_weight[nt_inds[i]] = span_range[1] - span_range[0]
        return(span_weight)

    def get_spanrange(self):
        """
        get the word spans of each non-terminal node based on the word order
        """
        matrix = self.delegators['csgraph'].tocsr()
        all_leaves = numpy.zeros(matrix.shape[0], dtype=numpy.bool_)
        all_leaves[numpy.logical_or(matrix.diagonal() == 0,
                   matrix.diagonal() > 1)] = True

        # retreive all non-terminals
        non_terms = self.visit_inodes_inorder()
        # locate leaves
        leaf_info = self.visit_leaves_inorder()

        map_leaf = {graph_idx: idx for idx, graph_idx in
                enumerate(map(itemgetter(0), leaf_info))}
        span_range = numpy.tile([len(map_leaf), 0], (len(non_terms), 1))
        # remove diagonal
        row_major = scipy.sparse.triu(matrix, k=1, format='csr') + \
            scipy.sparse.tril(matrix, k=-1, format='csr')
        col_major = scipy.sparse.triu(matrix, k=1, format='csc') + \
            scipy.sparse.tril(matrix, k=-1, format='csc')

        def update_ancestors(i, non_terms, span_range, col_major):
            parent = non_terms[i]
            while parent != 0:
                parent = col_major.indices[col_major.indptr[int(parent)]:
                        col_major.indptr[int(parent) + 1]]
                assert(parent != non_terms[i])
                # update min
                if span_range[i, 0] < span_range[non_terms == parent, 0]:
                    span_range[non_terms == parent, 0] = span_range[i, 0]
                # update max
                if span_range[i, 1] > span_range[non_terms == parent, 1]:
                    span_range[non_terms == parent, 1] = span_range[i, 1]

        for i in range(len(non_terms)):  # travel from bottom
            if not all_leaves[non_terms[i]]:  # checking if already updated
                children = row_major.indices[row_major.indptr[non_terms[i]]:
                        row_major.indptr[non_terms[i] + 1]]
                # 0 is left, 1 is right
                span_range[i][0] = numpy.min([map_leaf.get(children[0], numpy.inf),
                    span_range[i][0]])
                span_range[i][1] = numpy.max([map_leaf.get(children[1], 0),
                    span_range[i][1]])

                # also update its ancenstors
                update_ancestors(i, non_terms, span_range, col_major)
                # is reaching the leaf or all children are updated
                all_leaves[non_terms[i]] = numpy.all(all_leaves[children])

        all_spans = numpy.ma.masked_all((len(non_terms) + len(leaf_info), 2),
                                        dtype=numpy.int)
        all_spans[non_terms] = span_range
        all_spans[tuple(map_leaf.keys()), ] = numpy.asarray(
                list(map_leaf.values()))[:, numpy.newaxis]
        all_spans += numpy.tile([0, 1], (len(all_spans), 1))
        all_spans[tuple(map_leaf.keys()), ] = numpy.ma.masked
        return(all_spans)


class CSGraphProxy(Sequence, metaclass=abc.ABCMeta):
    """
    a proxy class to emulate scipy.sparse.spmatrix while providing the
    interface for the underlying delegators

    >>> proxy=CSGraphProxy(scipy.sparse.coo_matrix((10, 10),
    ... dtype=numpy.bool_), 'INDEXER', 'LABELS')
    >>> proxy.shape == (10, 10)
    True
    >>> (proxy.tocsr()).format
    'csr'
    >>> proxy.indexer
    'INDEXER'
    >>> proxy.labels
    'LABELS'
    >>> isinstance(proxy.csgraph, scipy.sparse.spmatrix)
    True
    >>> scipy.sparse.issparse(proxy)
    True
    """

    def __init__(self, csgraph, indexer, labels=None):
        descriptor = GraphProperty()
        descriptor['labels'] = labels
        self.delegate = Delegator(OrderedDict(
            [('csgraph', csgraph),
             ('indexer', indexer),
             ('descriptor', descriptor)]))


    def __getattribute__(self, name):
        delegate = object.__getattribute__(self, 'delegate')
        return(delegate.get_delegator(name))


    def __getitem__(self, key):
        """
        follows collections.Sequence protocol in order to emulate container
        behavior
        """
        delegate = object.__getattribute__(self, 'delegate')
        return(delegate.get_delegator('csgraph')[key])


    def __len__(self):
        """
        follows collections.Sequence protocol in order to emulate container
        behavior
        """
        delegate = object.__getattribute__(self, 'delegate')
        return(delegate.get_delegator('csgraph').shape[0])


class TreeIndexer(MutableMapping):

    def __init__(self):
        self._which_key = Indexer.tree
        self.index_pairs = (parray.array('i'), parray.array('i'))

    @property
    def which_key(self):
        return(self._which_key)

    def switch(self):
        self._which_key = Indexer(1 - self._which_key)

    def __delitem__(self, key):
        chosen_keys = set(self.index_pairs[self.which_key])
        if key in chosen_keys:
            self.index_pairs[self.which_key].remove(key)

    def __getitem__(self, key):
        return(self._get(key))

    def __iter__(self):
        return(iter(self.index_pairs[self.which_key]))

    def __len__(self):
        assert(len(self.index_pairs[0]) == len(self.index_pairs[1]))
        return(len(self.index_pairs[0]))

    def __setitem__(self, key, value):
        self._add(key, value)

    def _inverse_lookup(self, val):
        keys = list(self.keys())
        vals = [self.get(k) for k in keys]
        try:
            idx = vals.index(val)
        except:
            raise
        else:
            if idx == -1:
                raise KeyError('{!r} is not in table'.format(val))
            else:
                key = keys[idx]
        return(key)

    def _add(self, key, val, _disable=True):
        """
        translate the integer key into byte key and add to table
        """
        # byte_key = _translate(key)
        assert(key not in self.index_pairs[self.which_key])
        self.index_pairs[self.which_key].append(key)
        assert(val not in self.index_pairs[1 - self.which_key])
        self.index_pairs[1 - self.which_key].append(val)
        assert(len(self.index_pairs[0]) == len(self.index_pairs[1]))

    def _get(self, key):
        """
        translate the integer key into byte key and get the value
        """
        # byte_key = _translate(key)
        assert(self.index_pairs[self.which_key].count(key) == 1)
        byte_key = self.index_pairs[self.which_key].index(key)
        try:
            node_id = self.index_pairs[1 - self.which_key][byte_key]
        except KeyError:
            raise KeyError("key=%s [%s]" % (node_id, byte_key))
        return(node_id)

    def increment_if_none(self, key):
        """
        increment with one
        >>> indexer = TreeIndexer()
        >>> list(map(lambda i: indexer.increament_if_none(2 * i), range(10)))
        ...  == list(range(10))
        True
        >>> list(map(lambda i: indexer.increament_if_none(2 * i), range(10)))
        ...  == list(range(10))
        True
        """
        if key in self.index_pairs[self.which_key]:
            val = self._get(key)
        else:
            val = len(self.index_pairs[1 - self.which_key])
            self._add(key, val)
        return(val)


class GraphProperty(UserDict):

    def __init__(self, *args):
        super(__class__, self).__init__(*args)

    def __getattr__(self, key):
        return(self.data.get(key))


@get_phrases_helper.register(CSGraphProxy)
def get_phrases_from_matrix(first, graphs, vocab):
    phrase_to_node = {}

    for tree_idx in range(len(graphs)):
        all_spans = getattr(graphs[tree_idx], 'get_spanrange')()
        all_spans.mask = numpy.ma.nomask
        leaf_vocab = graphs[tree_idx].leaves()
        node_indices = graphs[tree_idx].treepositions()
        # forming complete sentence firstly and adding terminals
        terms = vocab[leaf_vocab]

        for i, word_span in enumerate(all_spans):
            phrase_key = ' '.join(terms[slice(*word_span)])
            phrase_to_node.setdefault(phrase_key, [])
            phrase_to_node[phrase_key].append(
                    (tree_idx, int(node_indices[i], base=2),
                        numpy.asscalar(numpy.diff(word_span))))
    return(phrase_to_node)


def _check_trees(tree, scipyecified_maxlen=-1):
    """
    ensure the tree is processed and return the proper max_len for constructing
    scipy.scipyarse.matrix
    """
    try:
        getattr(tree, '_index')
    except AttributeError:
        raise AttributeError('tree needs to be indexed')

    assert(numpy.all([type(leaf) in [int, numpy.intp] for leaf in
        tree.leaves()]))
    max_len = numpy.max([tree[pos[:-1]]._index for pos in
        tree.treepositions('leaf')]) + 1
    return(numpy.max([max_len, scipyecified_maxlen]))


def label_to_matrix(tree, max_len=-1, given_rootlabel=None):
    """
    get sentiment labels from  nodes in nltk.tree.Tree and form scipyarse array
    of the same dimensionality of tree_to_matrix
    """
    max_len = _check_trees(tree, max_len)
    label_mat = numpy.empty((max_len, max_len), dtype=numpy.int)
    for pos in tree.treepositions(order='preorder'):  # top down
        if isinstance(tree[pos], Tree):
            # to avoid zero label, all labels are added one
            label_mat[tree[pos]._index, tree[pos]._index] = \
                    tree[pos]._label + 1
    if given_rootlabel is not None:
        label_mat[tree._index, tree._index] = given_rootlabel
    return(label_mat)


def _check_format(matrix, format_='csr'):
    if hasattr(matrix, 'csgraph'):
        target = matrix.csgraph
    else:
        target = matrix

    if not scipy.sparse.issparse(matrix):
        target = scipy.sparse.csr_matrix(target)
    if matrix.format != format_:
        target = matrix.tocsr()
    return(target)


def to_csgraph(s, vocab, preprocessor=None, max_len=120):
    """
    turn the parsed tree as nested Node into scipy.sparse.csgraph

    @param tree is a namedtuple parsed by fromstring
    @param vocab using HashVectorizer to build the vocab dictionary or provided
               from the outer scope where vocab dictionary can be accumulated
               through parsing multiple trees
    @param max_level is the maximal level (usually is the length of leaves - 1)
                        which will yield the maximal index for the filled
                        balance binary tree as 2*(2**max_level-1)
    """
    if not vocab:
        raise ValueError("vocabulary cannot be empty, must be pre-computed")

    if isinstance(s, (str, bytes)):
        if type(s) == bytes:
            s = str(s, encoding='utf-8')
        tree = cnltk.fromstring(s)
    else:
        tree = s
    if preprocessor is None:
        preprocessor = lambda x: x.lower()
    csgraph = cnltk._check_allocation((max_len, max_len), sparse.lil_matrix,
            dtype=numpy.int32)
    labels = numpy.ma.masked_all(csgraph.shape[0], dtype=numpy.int8)
    bytes_table = TreeIndexer()
    queue = deque([(0, tree)])
    while len(queue) > 0:
        bit_id, cur_node = queue.pop()
        node_loc = bytes_table.increment_if_none(bit_id)
        for i, child in enumerate(cur_node.children):
            if isinstance(child, cnltk.Node):
                child_id = len(cur_node.children) * bit_id + i + 1
                child_loc = bytes_table.increment_if_none(child_id)
                csgraph[node_loc, child_loc] = 1
                csgraph[node_loc, node_loc] = 1
                queue.appendleft((child_id, child))
            else:
                child = preprocessor(child)
                if isinstance(child, list):
                #if hasattr(child, '__len__'):
                    assert(len(child) == 1)
                    child = child.pop()
                if child not in vocab: #unkown vocabulary
                    logger.debug("'%s' is unknown in current vocabulary set " 
                    "as unk" % child)
                    vocab_idx = vocab["-unk-"]
                else:
                    vocab_idx = vocab[child]
                if vocab_idx < 0: 
                    vocab_idx = len(vocab_idx)
                csgraph[node_loc, node_loc] = vocab_idx + 2
            labels[node_loc] = int(cur_node.label) # assign will not copy
                                                   # shared mask 
    return(CSGraphProxy(csgraph, bytes_table, labels))


def to_networkx(s, vocab, max_len=120, create_using=networkx.DiGraph):
    # mocking networkx constructor to nltk.tree.Tree
    proxy = to_csgraph(s, vocab, max_len=max_len)
    csgraph, indexer, labels = proxy.csgraph, proxy.indexer, proxy.labels

    # remove redundnat columns and rows
    nonzeros = numpy.arange(csgraph.shape[0], dtype=numpy.intp)[
            csgraph.diagonal() > 0]
    csgraph = csgraph[numpy.ix_(nonzeros, nonzeros)]
    # remove diagonal
    reduced_graph = scipy.sparse.triu(csgraph, k=1)  # root loc = 0
    if reduced_graph.nnz == 0:  # root loc = len(csgraph)
        reduced_graph = scipy.sparse.tril(csgraph, k=-1)
    assert(reduced_graph.nnz > 0)
    if sparse.issparse(reduced_graph) and reduced_graph.format != 'csr':
        reduced_graph = reduced_graph.tocsr()
    tree = create_using(reduced_graph)
    index2label, index2index, index2vocab = {}, {}, {}

    for i in tree.nodes():
        index2label[i] = labels[i]
        index2index[i] = indexer._inverse_lookup(i)
        if csgraph[i, i] > 1:  # leaves
            index2vocab[i] = csgraph[i, i] - 2
    networkx.set_node_attributes(tree, '_label', index2label)
    networkx.set_node_attributes(tree, '_index', index2index)
    networkx.set_node_attributes(tree, '_vocab', index2vocab)
    return(tree, indexer)


def dump_graphml(fn, g, attrnames, **kwargs):
    # convert attributes to string type
    networkx.write_graphml(g, fn, **kwargs)
