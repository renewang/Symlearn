from itertools import groupby
from io import StringIO

from sklearn.preprocessing import LabelBinarizer
from symlearn.csgraph import adjmatrix
from . import recursnn_utils
from .recursnn_utils import (visit_inodes_inorder,
        calculate_spanweight, get_spanrange)

import numpy as np
import scipy as sp
import logging

logger = logging.getLogger(__name__)

def computeNumericalGradient(costfunc, param, keyword=None, eps=1e-4):
    """
    A python version of computeNumericalGradient.m from Andrew Ng's Machine
    Learning Open Course (PA4).
    The description is from the computeNumericalGradient.m file:
    %COMPUTENUMERICALGRADIENT Computes the gradient using "finite differences"
    %and gives us a numerical estimate of the gradient.
    %   numgrad = COMPUTENUMERICALGRADIENT(J, theta) computes the numerical
    %   gradient of the function J around theta. Calling y = J(theta) should
    %   return the function value at theta.

    % Notes: The following code implements numerical gradient checking, and
    %        returns the numerical gradient.It sets numgrad(i) to (a numerical
    %        approximation of) the partial derivative of J with respect to the
    %        i-th input argument, evaluated at theta. (i.e., numgrad(i) should
    %        be the (approximately) the partial derivative of J with respect
    %        to theta(i).)
    %
    @params costfunc is a partial bound functions whose first two arguments
                          will bind precomputed tree and childrenmat
    @params learner is a recursive-auto-encoder instance

    potential example code:
    func = lambda x, y, score_func: score_func(x, y)
    cal_cost_obj = partial(func, X_sim, y_sim)
    numgrads = computeNumericalGradient(cal_cost_obj, logloss, classifier.W)
    assert(np.all(np.allclose(numgrads, analgrads))==True)
    """

    numgrad = np.zeros_like(param)
    perturb = np.zeros_like(param)
    for i, (p, n) in enumerate(zip(
            np.nditer(perturb, order='C', op_flags=['writeonly']),
            np.nditer(numgrad, order='C', op_flags=['writeonly']))):

        p[...] = eps
        # ensure the change is effect
        assert(param.ndim == 1 or perturb[i//perturb.shape[1],
                                          i % perturb.shape[1]] == eps)
        if keyword:
            # calculate cost function with parameter - purturb
            loss_rhs = costfunc(**{keyword: (param - perturb)})
            # calculate cost function with parameter + purturb
            loss_lhs = costfunc(**{keyword: (param + perturb)})
        else:
            loss_rhs = costfunc((param - perturb))
            loss_lhs = costfunc((param + perturb))

        # Compute Numerical Gradient
        n[...] = (loss_lhs - loss_rhs)/(2*eps)
        p[...] = 0.
        # ensure the change is effect
        assert(param.ndim == 1 or perturb[i//perturb.shape[1],
                                          i % perturb.shape[1]] == 0.)
        return(numgrad)


def label_binarize(y_sim, *args, **kwargs):
    """
    @param y_sim is y label either in 1D or on-hot-coding
    """
    if np.all(np.unique(y_sim) == np.arange(2)):  # binary cases
        # TODO: change to sklearn.preprocessing.label_binarize, and given
        # classes = 2
        y = np.vstack([1-y_sim, y_sim]).T
    else:
        y = LabelBinarizer().fit_transform(y_sim)
    return(y)


class Memoize():
    """
    Memoize(fn) - an instance which acts like fn but memoizes its arguments
    Will only work on functions with non-mutable arguments
    Borrow from memorizing recipe: http://code.activestate.com/recipes/52201/
    """
    def __init__(self, fn):
        self.fn = fn
        self.memo = {}

    def __call__(self, keyword, *args, index=-1):
        key = self._compute_hash_key(*args)
        if key not in self.memo:
            self.memo[key] = self.fn(*args)
        if hasattr(self.memo[key], keyword):
            return getattr(self.memo[key], keyword)[index]
        else:
            return None

    def _compute_hash_key(self, *args):
        strbuf = StringIO()
        for a in args:
            strbuf.writelines(str(a))
        return(hash(strbuf.getvalue()))


def optimize_wrapper(wrapped_func, newshape):
    """
    to return a new function which can take 1D as input and then pass to the
    wrapped_func for actual calculation
    @param wrapped_func is a function receives reshape arguments by
                        optimize_wrapper function
    @param newshape is a new shape in tuple passing to wrapped_func
    """
    def wrapper(param, *args, **kwargs):
        # the required transformed here
        param_t = param.reshape(newshape)
        result = wrapped_func(param_t, *args, **kwargs)
        return(result.ravel())
    return(wrapper)


def modify_partial(func, *args):
    """
    make partial to bind 1st parameter
    @param func is the function object whose 1st paramter will be pre-bound
    @param args are the rest arguments will be passed into func
    """
    def bind_first_arg(x):
        res = func(x, *args)
        return(res)
    return(bind_first_arg)


def activation_gradient(activation, X):
    """
    manually calculate f'(x)
    @param activation is a function object which serves as activationi function
                      in neural network. Either is tanh or expit
    @param X is the input numpy array
    """
    z = 1.0
    w = 1.0
    if activation.__name__ == 'tanh':
        z = 2.0
        w = 4.0
    elif activation.__name__ != 'expit':
        raise NotImplementedError

    return(w*sp.special.expit(z*X)*(1-sp.special.expit(z*X)))


def compute_len_stats(len_dist):
    """
    return lengths distribution of sentence plus cumulative sum
    @param len_dist is a list whose elements are corresponding sentence length
    """
    # TODO: consider to replace with numpy.bincount
    len_count = np.zeros(len(np.unique(len_dist)) + 1, dtype=np.int)
    for i, cur_len in enumerate(np.unique(len_dist)):
        len_count[i + 1] = np.sum(len_dist == cur_len)
    return(np.unique(len_dist), len_count)


def _iter_samples(word2index, embedding, **kwargs):
    """
    prepare sample for recursive trees-based training
    @param word2index is a list/OrderedDict/numpy.ndarray which each entry
                      holds the word order of sentence
    @param embedding is a word embedding matrix used for construct input word
                      vectors
    """

    params_name = [
        'input_vectors',
        'traversal_order',
        'inverse_inputs',
        'parent_children_mapping',
        'true_output',
        'span_weight']

    params_types = [list] * 6
    hyper_params = {'error_weight': 0.1, 'alpha': 0.01,
                    'classification_threshold': 0.5, 'corruption_level': 0.}

    # TODO: improve this code
    for param_name in hyper_params.keys():
        # TODO: check out argparser in python std
        if param_name in kwargs:
            logger.debug(
                "hyperparameters {} gets updated to {:.3f} (default:"
                "{:.3f})".format(param_name,
                                 kwargs[param_name], hyper_params[param_name]))
            hyper_params[param_name] = kwargs[param_name]

    if 'true_output' not in kwargs:
        params_name.remove('true_output')
        true_output = None
    else:
        true_output = kwargs['true_output']

    if kwargs.get('trees'):
        trees = kwargs['trees']
        iter_trees = iter(trees)
    else:
        trees = None
        iter_trees = None

    if kwargs.get('max_len'):
        max_len = int(kwargs['max_len'])
    else:
        max_len = np.max(list(map(len, word2index)))

    examples = []
    index2len = []
    iter_over_word2index = groupby(word2index, key=len)
    start = 0
    for cur_len, word_orders in iter_over_word2index:
        if cur_len > max_len:
            break
        # allocate new list
        examples.append({})
        tree_size = 2 * cur_len - 1
        for key, cls in zip(params_name, params_types):
            examples[-1][key] = cls()
        for word_order in word_orders:
            if not trees and kwargs.get('autoencoder'):
                est_gtree, est_cost = recursnn_utils.build_greedy_tree(
                    word_order, embedding=embedding,
                    autoencoder=kwargs.get('autoencoder'))
            else:
                try:
                    est_gtree = next(iter_trees)
                except StopIteration:
                    if cur_len != max_len:
                        raise RuntimeError
                    else:
                        pass
                except:
                    raise
            vocab = recursnn_utils.embed_words(word_order, embedding)
            est_gtinfo = recursnn_utils.preprocess_input(est_gtree, vocab)
            assert(
                np.all(np.linalg.norm(est_gtinfo[2][np.sum(
                    est_gtinfo[2], axis=1) > 0], 2, axis=1)) == 1.0)
            if kwargs.get('span_weight'):
                span_weight = est_gtinfo[3]
            else:
                span_weight = 0.5 * np.ones_like(est_gtinfo[3])
            if isinstance(true_output, np.ndarray):
                assignments = [est_gtinfo[2], est_gtinfo[0],
                               np.zeros_like(est_gtinfo[2]), est_gtinfo[1],
                               true_output[start:start + tree_size],
                               span_weight]
                start += tree_size
            else:
                assignments = [est_gtinfo[2], est_gtinfo[0],
                               np.zeros_like(est_gtinfo[2]), est_gtinfo[1],
                               span_weight]

            for key, val in zip(params_name, assignments):
                examples[-1][key].append(val)
        sample_size = len(examples[-1]['input_vectors'])
        for key, val in examples[-1].items():
            if isinstance(val, list):
                # first, combine the matrices for all samples
                examples[-1][key] = np.concatenate(val, axis=0)
                assert(examples[-1][key].ndim <= 2)
            if key == 'traversal_order':
                examples[-1][key] += np.repeat(np.arange(sample_size) *
                                               tree_size, cur_len - 1)
            elif key == 'parent_children_mapping':
                examples[-1][key] += (
                        np.repeat(np.arange(sample_size) *
                                  tree_size, tree_size))[:, np.newaxis]
            elif key == 'true_output':
                if examples[-1][key].ndim == 2:
                    examples[-1][key] = np.argmax(examples[-1][key], axis=1)
        index2len.append(cur_len)
        examples[-1].update(hyper_params)
        examples[-1]['sample_size'] = sample_size
    return((np.array(index2len), examples))


def _combine_sample(examples):
    """
    combine sample produce by _iter_sample which are grouped by lengths into
    one chunk
    """
    combined_example = {}
    for example in examples:
        for key, value in example.items():
            ent = combined_example.setdefault(key, [])
            ent.append(value)
    assert(np.all(np.asarray(list(map(len, combined_example.values()))) ==
                  len(examples)))
    return(combined_example)

def _iter_matrix_groups(matrix, **kwargs):
    if 'embedding' in kwargs and 'vocab_size' in kwargs:
        raise RuntimeError(
        'embedding and vocab_size cannot be specified at the same time') 
    elif 'embedding' not in kwargs and 'vocab_size' not in kwargs:
        raise RuntimeError(
        'embedding and vocab_size cannot be specified at the same time') 

    embedding = kwargs.get('embedding', None)
    if hasattr(embedding, 'components_'):
        embedding = kwargs.get('embedding').components_.T

    vocab_size = kwargs.get('vocab_size', None) or embedding.shape[0]
    if kwargs.get('traverse_func'):
        traverse_func = kwargs.get('traverse_func')
    else:
        traverse_func = visit_inodes_inorder

    mapping = recursnn_utils.retrieve_mapping(matrix)
    span_range = get_spanrange(matrix)
    traversal_order = traverse_func(matrix)

    # checking if indexing in mapping are attainable (not heap-ordered)
    if np.max(mapping) >= len(mapping):
        remap_lookup = np.ma.MaskedArray(np.zeros(matrix.shape[0],
            dtype=np.intp), mask=matrix.diagonal() == 0)
        remap_lookup[~remap_lookup.mask] = np.arange(np.sum(
            [matrix.diagonal() > 0]))
        for row in np.nditer(mapping, op_flags=['readwrite'],
                flags=['external_loop']):
            row[...] = remap_lookup[row]
        for cols in np.nditer(traversal_order, op_flags=['readwrite'],
                flags=['external_loop']):
            cols[...] = remap_lookup[cols]
    
    if kwargs.get('return_weight', False):
        span_weight = calculate_spanweight(matrix, span_range)
        node_weight = np.nan_to_num(
                np.clip(span_weight, 1, np.inf)[mapping] / np.sum(
                    np.clip(span_weight, 1, np.inf)[mapping], axis=1,
                    keepdims=True))
    else:
        node_weight = span_range

    if embedding is None:
        # return indices
        word_mat = recursnn_utils.get_wordindices(matrix, vocab_size,
                node_weight)
    else:
        word_mat = recursnn_utils.get_wordmatrix(matrix, embedding)
    return(traversal_order, word_mat, node_weight, mapping)
