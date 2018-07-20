from pathlib import Path
from functools import reduce
from enum import IntEnum
from matplotlib import gridspec
from collections import namedtuple
from contextlib import contextmanager
from collections import Iterable
from functools import partial
from operator import itemgetter

from symlearn.csgraph.sampling import permutate_sampling
from symlearn.csgraph.sampling import load_and_modify
from symlearn.csgraph.sampling import bootstrap_sampling
from symlearn.csgraph import cnltk_trees as cnltk
from symlearn.utils import run_batch

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import logging
import os
import re


__all__ = ['run_analysis', 'unmask_results', 'run_permute']

logging.basicConfig(format='%(asctime)s %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)

seed = 42
rng = np.random.RandomState(seed)
data_dir = os.path.join(os.getenv('WORKSPACE'), 'Kaggle/symlearn/data/')
keepdims_sum = partial(np.apply_over_axes, np.sum)
sent_colors = ['#98AFC7', '#3090C7', '#48CC7D', '#E47451', '#E77471']


def print_var(self):
    return(self.name)

VARS = IntEnum('VARS', 'level root children')
VARS.__str__ = print_var


def wrapper_func(func, *bound_arg, **kwargs):
    def helper_func(x, **kwargs):
        return(itemgetter(0)(partial(func, *bound_arg)(x, **kwargs)))
    return(helper_func)


def cal_univariate_prob(var_in_interest, merge_result, eps=1e-15):
    assert(merge_result.ndim == 4)
    joint_probs = np.clip(np.nan_to_num(merge_result /
        keepdims_sum(merge_result, tuple(VARS))), eps, 1 - eps)
    sumup_axes = set(list(VARS)) - set([var_in_interest])
    probs = keepdims_sum(joint_probs, tuple(sumup_axes))
    return(probs)


def cal_joint_prob(merge_result, eps=1e-15):
    assert(merge_result.ndim == 4)
    joint_probs = np.clip(np.nan_to_num(merge_result /
       keepdims_sum(merge_result, tuple(VARS))), eps, 1 - eps)
    return(joint_probs)


def cal_conditional_prob(var_in_interest, merge_result, eps=1e-15):
    assert(merge_result.ndim == 4)
    assert(isinstance(var_in_interest, Iterable))
    joint_probs = cal_joint_prob(merge_result, eps)
    sumup_axes = set(list(VARS)) - set(var_in_interest)
    conditional_probs = np.clip(np.nan_to_num(joint_probs /
        keepdims_sum(joint_probs, tuple(sumup_axes))), eps, 1 - eps)
    return(conditional_probs, joint_probs)


def cal_marginal_prob(var_in_interest, marginal_var, merge_result, eps=1e-15):
    assert(merge_result.ndim == 4)
    sumup_stat = keepdims_sum(merge_result, marginal_var)
    return(cal_conditional_prob(var_in_interest, sumup_stat))


def cal_self_mi(var_in_interest, merge_result, eps=1e-15):
    """
    compute entropy (self mutual information) for single variable where entropy
    is defined as the expectation of -log(p(x)); therefore
    :math: \sum_x -log(p(x))p(x)

    Parameters
    ----------
    """
    assert(merge_result.ndim == 4)
    probs = cal_univariate_prob(var_in_interest, merge_result, eps)
    return(-1 * probs * np.log(probs))


def cal_conditional_entropy(var_in_interest, merge_result, eps=1e-15):
    """
    using ...math:: H(Y|X) &= \sum_x p(x) \sum_y -log(p(y|x))p(y|x)
                           &= \sum_x\sum_y -log(p(y|x))p(x, y)
    Parameters
    ----------
    """
    assert(merge_result.ndim == 4)
    if not isinstance(var_in_interest, Iterable):
        var_in_interest = [var_in_interest]
    conditional_probs, joint_probs = cal_conditional_prob(var_in_interest,
            merge_result, eps)
    return(-1 * joint_probs * np.log(conditional_probs))


def cal_marginal_entropy(var_in_interest, marginal_var, merge_result,
        eps=1e-15):
    """
    using ...math:: H(Y|X) &= \sum_x p(x) \sum_y -log(p(y|x))p(y|x)
                           &= \sum_x\sum_y -log(p(y|x))p(x, y)
    Parameters
    ----------
    """
    assert(merge_result.ndim == 4)
    if not isinstance(var_in_interest, Iterable):
        var_in_interest = [var_in_interest]
    conditional_probs, joint_probs = cal_marginal_prob(var_in_interest,
            marginal_var, merge_result, eps)
    return(-1 * joint_probs * np.log(conditional_probs))


def cal_joint_entropy(merge_result, eps=1e-15):
    """
    using ...math:: H(X, Y) = \sum_x\sum_y -log(p(x, y))p(x, y)

    Parameters
    ----------
    """
    assert(merge_result.ndim == 4)
    joint_probs = cal_joint_prob(merge_result, eps)
    return(-1 * joint_probs * np.log(joint_probs))


def get_entropy(stat_result, entropy_funcs, axis_to_level=0, cutoff=6,
        eps=1e-15, init_func=np.ones_like):
    """
    Parameters
    ----------
    @param stat_result is a 3-ways contigency table for level (axis_to_level),
    sentiment label of root (axis_to_level + 1) and of children (axis_to_level
    + 2)
    @param entropy_funcs is collection of callable object defined in
    cal_joint_entropy, cal_conditional_entropy and cal_self_mi with prebound
    arguments
    @cutoff is an integer indicating the least counts each cell should have
    @eps is a floating number will be used to avoid -inf when applying log to
    zero
    @init_func is a function to apply constant pseudo count on all entries
    @retuns point conditioal entropy
    """
    stat_result = merge_levels(stat_result, axis_to_level=axis_to_level,
        cutoff=cutoff)
    stat_result += init_func(stat_result)
    if axis_to_level == 0:
        stat_result = np.expand_dims(stat_result, axis=0)
    if not isinstance(entropy_funcs, Iterable):
        entropy_funcs = [entropy_funcs]
    results = [entropy_func(stat_result) for entropy_func in entropy_funcs]
    logging.debug("params used for cal_mutual_entropy:init_func={init_func};"
                 "axis_to_level={axis_to_level};cutoff={cutoff};".format(
                     init_func=init_func.__name__, axis_to_level=axis_to_level,
                     cutoff=cutoff))
    return(results)


def get_divergence(merge_result):
    assert(merge_result.ndim == 4)
    name_of_dist = {(VARS.root, VARS.children):
            [('%s, %s' % (VARS.root, VARS.children),
                wrapper_func(cal_conditional_prob, [VARS.level]))],
            (VARS.root,): [(str(VARS.root),
                wrapper_func(cal_marginal_prob, [VARS.level], VARS.children)),
            ('%s | %s' % (VARS.children, VARS.root),
                wrapper_func(cal_conditional_prob, [VARS.root, VARS.level]))],
            (VARS.children,): [(str(VARS.children),
                wrapper_func(cal_marginal_prob, [VARS.level], VARS.root)),
            ('%s | %s' % (VARS.root, VARS.children),
                wrapper_func(cal_conditional_prob, [VARS.children, VARS.level])
             )]}
    divergence = {}
    for var, funcs in name_of_dist.items():
        divergence.setdefault(var, [])
        for name, func in funcs:
            divergence[var].append((name, np.mean(
                (cal_divergence(func, merge_result).sum(
                    axis=(-2, -1))), axis=0)))
    return(divergence)


def merge_levels(stat_result, cutoff=6, axis_to_level=0):
    """
    merge the levels with count lower than the specified cutoff by using mask
    Parameters
    ----------
    @param stat_result is a numpy ndarray storing the sentiment distribution
                       for each node conditioning on root label
    @param cutoff is the threshold to decide if the level has too few samplng
                    and must be merged into lower levels
    @param axis_to_level is the pre-calculated max_level to avoid confusion
    @return numpy ndarray with merged result


    >>> test_mat = np.concatenate([100 * np.ones((100, 3, 2, 2)),
    ... 4 * np.ones((100, 2, 2, 2))], axis=1)
    >>> exp_mat = np.concatenate([test_mat[:, :2, :],
    ... np.sum(test_mat[:, -3: :], axis=1, keepdims=True)], axis=1)
    >>> np.all(merge_levels(test_mat, axis_to_level=1, cutoff=6).compressed(
    ... ).reshape(100, -1, 2, 2) == exp_mat)
    True
    >>> np.all(test_mat == np.concatenate([100 * np.ones((100, 3, 2, 2)),
    ... 4 * np.ones((100, 2, 2, 2))], axis=1))
    True
    >>> test_mat = np.concatenate([100 * np.ones((100, 3, 2, 2)),
    ... np.zeros((100, 2, 2, 2))], axis=1)
    >>> np.all(merge_levels(test_mat, axis_to_level=1, cutoff=6).compressed(
    ... ).reshape(100, -1, 2, 2) == test_mat[:, :3, :])
    True
    >>> np.all(test_mat == np.concatenate([100 * np.ones((100, 3, 2, 2)),
    ... np.zeros((100, 2, 2, 2))], axis=1))
    True
    >>> test_mat = np.arange(100, 0, -1)[:, np.newaxis] * np.ones((100, 5))
    >>> exp_mat = np.vstack([test_mat[:-11], np.sum(test_mat[-11:], axis=0)])
    >>> np.all(merge_levels(test_mat, cutoff=11).compressed().reshape(-1, 5)
    ... == exp_mat)
    True
    >>> np.all(test_mat == np.arange(100, 0, -1)[:, np.newaxis] * np.ones(
    ... (100, 5)))
    True
    >>> test_mat = np.arange(2 * 2 * 5, 0, -1).reshape((5, 2, 2))
    >>> exp_mat = np.concatenate([test_mat[:2, :], np.sum(test_mat[2:, :],
    ... axis=0, keepdims=True)], axis=0)
    >>> np.all(merge_levels(test_mat, cutoff=11).compressed().reshape(-1, 2, 2)
    ... == exp_mat)
    True
    >>> np.all(test_mat == np.arange(2 * 2 * 5, 0, -1).reshape((5, 2, 2)))
    True
    >>> test_mat = 100 * np.ones((100, 3, 2, 2))
    >>> np.all(merge_levels(test_mat, axis_to_level=1, cutoff=6).compressed(
    ... ).reshape(100, -1, 2, 2) == test_mat)
    True
    >>> merge_levels([[6, 6, 9], [2, 1, 0], [7, 8, 9], [1, 2, 3], [4, 2, 7]],
    ... cutoff=5)
    Traceback (most recent call last):
        ...
    RuntimeError: cannot handle not consecutive merge
    """
    if not isinstance(stat_result, np.ndarray):
        stat_result = np.asarray(stat_result)
    max_level = stat_result.shape[axis_to_level]
    # taking flattened view after the level axis
    merged = stat_result.reshape(stat_result.shape[:axis_to_level + 1] + (-1,))
    merged = np.ma.MaskedArray(merged, mask=True)
    swapped = merged
    if axis_to_level != 0:
        swapped = swapped.swapaxes(0, axis_to_level)

    # create mask for the candidate levels for merging
    mask = np.all(merged.data < cutoff, axis=axis_to_level + 1)
    assert(mask.shape == merged.shape[:-1])
    # merge levels needs to be greater than the maximal level - 1
    if mask.sum() == 0 or np.all(
            mask.sum(axis=axis_to_level) > (max_level - 1)):
        logging.debug("merge_level (cutoff={cutoff}) "
                      "has not been conducted".format(cutoff=cutoff))
        return(stat_result.view(np.ma.MaskedArray))
    else:
        merge_loc = np.zeros(mask.shape[axis_to_level], dtype=mask.dtype)
        for m, loc in np.nditer([mask, merge_loc],
                op_flags=[['readonly'], ['readwrite']],
                flags=['reduce_ok', 'external_loop', 'buffered']):
            # using xor to find the first merging point
            pm = np.pad(m, (1, 0), mode='edge')[:-1]
            loc[...] |= np.logical_xor(m, pm)
        if np.sum(merge_loc) > 1:
            merge_max = np.max(np.arange(len(merge_loc)) * merge_loc)
            merge_loc[...] = False
            merge_loc[merge_max] = True
            # raise RuntimeError("cannot handle not consecutive merge")

        # set mask
        merged_level = np.arange(len(merge_loc))[merge_loc] - 1
        combined = np.vstack([swapped[:merged_level],
            swapped.data[merged_level:].sum(axis=0)[np.newaxis, :],
            np.zeros((max_level - merged_level - 1, ) + swapped.shape[1:],
                dtype=swapped.dtype)])
        combined[merged_level + 1:] = np.ma.masked
        if axis_to_level != 0:
            combined = combined.swapaxes(0, axis_to_level)
        return(combined.reshape(stat_result.shape))


def normalized_mi(normed_vars, merge_result, eps=1e-15):
    """
    calculate the normalized mutual entropy by the bound of sqrt(H(x)H(y))
    Parameters
    ----------
    """
    assert(len(normed_vars) > 1)
    entropies = [cal_self_mi(var, merge_result, eps) for var in normed_vars]
    return(np.sqrt(reduce(np.multiply, entropies)))


def cal_normed_mi(var_in_interest, merge_result, normed_func=None, eps=1e-15):
    """
    calculate mutual entropy conditioning on the root node

    Parameters
    ----------

    >>> levels, joints, children, parents = 0.1 * np.arange(1, 5), np.tile(
    ... [[0.4, 0.1], [0.2, 0.3]], (4, 1, 1)), [0.6, 0.4] * np.ones(
    ...  (4, 1, 1)), 0.5 * np.ones((4, 2, 1))
    >>> counts = 100 * np.arange(1, 5)[:, np.newaxis, np.newaxis] * joints
    >>> normed_mutuals = cal_mutual_entropy(counts, init_func=np.zeros_like)
    >>> expect = levels[:, np.newaxis, np.newaxis] * joints * (
    ... np.log(joints) - np.log(parents) - np.log(children))
    >>> np.allclose(expect, normed_mutuals)
    True
    """
    if not isinstance(var_in_interest, Iterable):
        var_in_interest = [var_in_interest]
    mutuals = cal_mi(var_in_interest, merge_result, eps=eps)
    if normed_func is None:
        normed_func = normalized_mi
    normed_vars = set(list(VARS)) - set(var_in_interest)
    normed_factor = normed_func(normed_vars, merge_result, eps)
    assert(np.all(np.isfinite(normed_factor)))
    normed_mutuals = mutuals / normed_factor
    assert(np.all(np.abs(normed_mutuals <= 1)))
    return(mutuals, normed_mutuals)


def cal_point_mi(var_in_interest, merge_result, eps=1e-15):
    """
    calculate the point-wise conditional mutual information in e base
    parameters given each level

    .. math:: I(x, y) = p(x, y)\frac{log(p(x, y))}{log(p(x))log(p(y))}
    point_mi is \frac{log(p(x, y))}{log(p(x))log(p(y))}

    Parameters
    ----------
    @var_in_interest is a collection of instances of VARS on whic the joint
    probability will conditioning on.
    @param merge_result is the contigency table group by the level
    @eps is the numerical value to represent the exteme minum value in order to
            get infinite value when taking log
    """

    assert(merge_result.ndim == 4)
    if not isinstance(var_in_interest, Iterable):
        var_in_interest = [var_in_interest]
    sumup_axes = set(list(VARS)) - set(var_in_interest)
    # conditional joint probability given var_in_interest
    joint_probs = np.clip(np.nan_to_num(merge_result /
            keepdims_sum(merge_result, tuple(sumup_axes))),
            eps, 1.0 - eps)
    # compute marginal probability of rows (axis=1) and columns (axis=0)
    marginal_probs = np.empty(2, dtype=np.object)
    for i, sumup_axis in enumerate(sumup_axes):
        marginal_probs[i] = np.clip(np.nan_to_num(
            keepdims_sum(merge_result, sumup_axis) /
            keepdims_sum(merge_result, tuple(sumup_axes))),
            eps, 1.0 - eps)
    mutuals = np.log(joint_probs) - reduce(np.add, map(np.log,
        marginal_probs))
    return(mutuals)


def cal_mi(var_in_interest, merge_result, eps=1e-15):
    assert(merge_result.ndim == 4)
    mutuals = cal_point_mi(VARS.level, merge_result, eps=eps)
    joint_probs = merge_result / keepdims_sum(merge_result, tuple(VARS))
    # weighted by joint probability
    return(joint_probs * mutuals)


def cal_divergence(prob_func, merge_result, split=[1], eps=1e-15):
    # raw as observing model and rep as null model
    raw, rep = np.split(merge_result, split, axis=0)
    weight = cal_joint_prob(raw, eps)
    divergence = [prob_func(sample, eps=eps) for sample in [raw, rep]]
    return(weight * np.log(reduce(np.true_divide, divergence)))


def stat_handler(update_res, memmap_fp, memmap_filename, **kwargs):
    if memmap_fp is None:
        memmap_fp = np.memmap(memmap_filename, **kwargs)
    if update_res.size > memmap_fp.size:
        memmap_fp.flush()
        raise ValueError('mmap size is not big enough')
    elif update_res.size < memmap_fp.size:
        update_res.resize(memmap_fp.shape, refcheck=False)
    return(memmap_fp)


def tree_handler(update_res, memmap_fp, memmap_filename, **kwargs):
    if memmap_fp is None:
        memmap_fp = np.memmap(memmap_filename, mode='w+',
                shape=update_res.shape, **kwargs)
    else:
        memmap_fp = np.memmap(memmap_filename, mode='r+',
                shape=update_res.shape, **kwargs)
    return(memmap_fp)


def _check_result(result, expects):
    val, _, data = np.unique(expects, return_index=True, return_counts=True)
    idx = val.astype(np.intp)
    if np.all(idx - 1 >= 0):
        idx -= 1
    return(np.all(result[idx] == data))


def load_batches(stat_file):
    # loading batch results
    iscorrect = False
    with np.load(stat_file.as_posix(), mmap_mode='r') as fd:
        if 'labels' not in fd:
            iscorrect = True
            stat_res, levels, labels = fd['stat'], fd['levels'], fd['lables']
        else:
            stat_res, levels, labels = fd['stat'], fd['levels'], fd['labels']

    if iscorrect:
        with open(stat_file.as_posix(), mode='wb') as fp:
            np.savez(fp, stat=stat_res, labels=labels, levels=levels)
    return(stat_res, levels, labels)


def post_process(levels, labels, max_level):
    tree_res = np.concatenate([np.expand_dims(levels, axis=1),
        np.expand_dims(labels, axis=1)], axis=1).astype(np.int)
    tree_res[tree_res < 0] = 0  # correct deprecated padding -1
    if tree_res.shape[-1] < max_level:
        pad_res = np.pad(tree_res, ((0, 0), (0, 0),
            (0, max_level - tree_res.shape[-1])),
            mode='constant', constant_values=0)
        assert(np.all(pad_res >= 0))
    return(pad_res)


def loadstat_handler(i, stat_file, vocab, **kwargs):
    max_level = kwargs.get('max_level', 120)
    n_classes = kwargs.get('n_classes', 5)
    offset_ = kwargs.get('offset_', 0)
    size_count = kwargs.get('size_count', 0)
    stat_result = kwargs.get('stat_result', None)
    tree_result = kwargs.get('tree_result', None)
    treefilename = kwargs.get('treefilename', None)
    statfilename = kwargs.get('statfilename', None)
    logging.info('start handling {:s}'.format(stat_file.as_posix()))
    n_batch = re.findall(r'\w+\.([0-9]+)\.npz', stat_file.name)
    assert(len(n_batch) == 1)
    n_batch = int(n_batch.pop())
    stat_res, levels, labels = load_batches(stat_file)
    # checking
    assert(levels.dtype == labels.dtype)
    assert(levels.shape == labels.shape)
    assert(stat_res is not None and stat_res.shape == (max_level,
        n_classes, n_classes))
    # ensure all levels are correct
    assert(_check_result(stat_res.sum(axis=(1, -1)), levels[levels > 0]))
    # ensure all node labels are correct
    assert(_check_result(stat_res.sum(axis=(0, 1)), labels[levels > 0]))

    # combine multiple npz into separate memmap npy
    stat_result = load_and_modify(statfilename, stat_res, stat_handler,
            stat_result, dtype=stat_res.dtype, mode='w+',
            shape=(max_level, n_classes, n_classes))
    stat_result.flush()
    logging.info('cumulative {:d} are stored in {:s}'.format(n_batch,
        statfilename))
    # read with offset and append along axis=0, expand shape[1]
    # (treesize) if needs
    tree_res = post_process(levels, labels, max_level)
    tree_result = load_and_modify(treefilename, tree_res, tree_handler,
            tree_result, dtype=np.int, offset=offset_)
    tree_result.flush()
    offset_ += tree_result.nbytes
    size_count += len(tree_result)
    logging.info('cumulative {:d} (size={:d}) are stored in {:s}'.format(
        n_batch, size_count, treefilename))

    # ensure all levels are correct
    tree_top = np.memmap(treefilename, mode='r', shape=(size_count,) +
            tree_result.shape[1:], dtype=np.int)
    assert(_check_result(stat_result.sum(axis=(1, -1)),
        tree_top[:, 0, :][tree_top[:, 0, :] > 0]))
    # ensure all node labels are correct
    assert(_check_result(stat_result.sum(axis=(0, 1)),
        tree_top[:, 1, :][tree_top[:, 0, :] > 0]))
    del tree_top


def calstat_handler(i, trees, vocab, **kwargs):
    max_level = kwargs.get('max_level', 120)
    n_classes = kwargs.get('n_classes', 5)
    results = [cnltk.to_networkx(str(tree), vocab, max_level) for tree in
            trees]
    stat_result, tree_result = cnltk.compute_stat_result(results,
            max_level, n_classes)
    levels, labels = cnltk.postprocess_results(tree_result)
    with open(os.path.join(data_dir, 'stat_result.%d.npz' % i), 'wb') as fp:
        np.savez(fp, stat=stat_result, labels=labels, levels=levels)


def unmask_results(results):
    """
    remove the mask and split the sample from resample which are satcked
    togehter for computing merge_level result

    Parameters
    ----------
    @param results is a list which stores numpy.ma.MaskedArray masking by
    merged_level result
    @returns a split list of two multi-dimensional histograms without masking
    for sample and resampling.
    """
    stats = []
    for res in results:
        assert(hasattr(res, 'mask'))
        compressed = res.compressed().reshape(*(
            -1 if i == 1 else shape_i for i, shape_i in enumerate(res.shape)))
        stats.append(np.split(compressed, [1], axis=0))
    return(stats)


@contextmanager
def open_result(result_fn=None):
    """
    open resampling result in npz format computed by run_bootstrap_result

    Parameters
    ----------
    @param result_fn is the file name stored resampling result and will
    combined with data_dir
    @returns a open npz file handle used under context manager
    """
    if result_fn is None:
        result_fn = os.path.join(data_dir, 'kl_divergence.npz')
    else:
        result_fn = os.path.join(data_dir, result_fn)
    result = np.load(result_fn)
    logging.info("open {:s} for computation".format(result_fn))
    yield result
    result.close()


@contextmanager
def open_stat(stat_fn=None, **kwargs):
    """
    open observed sample result in npz format computed by run_stat_result
    Usage:

    with open_stat(stat_fn='tree_result.npy', shape=(11855, 2, 120),
                   dtype=np.int) as tree_result:
        levels, labels = tree_result[:, 0, :], tree_result[:, 1, :]
        targets = labels[:, 0]
        masked_levels = np.ma.masked_equal(levels, 0, copy=False)
        masked_labels = np.ma.MaskedArray(labels, masked_levels.mask, copy=False)

    Parameters
    ----------
    @param stat_fn is the file name stored resampling result and will combined
    with data_dir
    @returns a open npz file handle used under context manager
    """
    if stat_fn is None:
        stat_fn = os.path.join(data_dir, 'stat_result.npy')
    else:
        stat_fn = os.path.join(data_dir, stat_fn)
    memmap_kwargs = {}
    memmap_kwargs.setdefault('shape', (120, 5, 5))
    memmap_kwargs.setdefault('dtype', np.float)
    memmap_kwargs.update(kwargs)
    stat_result = np.memmap(stat_fn, mode='c', **memmap_kwargs)
    logging.info("open {:s} for computation".format(stat_fn))
    yield stat_result
    del stat_result


def proc_stat(result_fn=None, stat_fn=None, **kwargs):
    """
    a convient function called by the client to conduct basic process on sample
    and resample data including combine two samples and merge insufficient
    counts

    Parameters
    ----------
    @param stat_fn is the file name stored resampling result and will combined
    with data_dir
    @returns a open npz file handle used under context manager
    """
    cutoff = kwargs.get('cutoff', 6)
    axis_to_level = kwargs.get('axis_to_level', 1)
    with open_result(result_fn) as result, open_stat(None) as stat_result:
        resamples = result['resamples']
        n_classes = resamples.shape[2]
        input_ = np.concatenate([stat_result[np.newaxis, :], resamples])
        merged_result = merge_levels(input_, axis_to_level=axis_to_level,
                cutoff=cutoff).compressed().reshape(
                input_.shape[0], -1, n_classes, n_classes)
    return(merged_result)


def _default_statsfuncs():
    entropy_funcs = []
    # calcualte conditional entropy P(VARS.children|VARS.root, VARS.level)
    entropy_funcs.append(partial(cal_conditional_entropy, [VARS.root,
        VARS.level]))
    # calcualte conditional entropy P(VARS.root|VARS.children, VARS.level)
    entropy_funcs.append(partial(cal_conditional_entropy, [VARS.children,
        VARS.level]))
    # normalize point-wise
    entropy_funcs.append(partial(cal_normed_mi, [VARS.level]))
    return(entropy_funcs)


def run_analysis(result_fn=None, stat_fn=None, **kwargs):
    """
    a convient function called by the client to conduct conditional entropy
    computation on sample and resample data

    Parameters
    ----------
    @param result_fn is the file name will pass along to open_result
    @param stat_fn is the file name will pass along to open_stat
    @returns a namedtuple has fields as following:
    cond_parents: conditional entropy of P(children|parent)
    cond_children: conditional entropy of P(parent|children)
    mi: mutual information without normalized pointwise
    nmi: pointwise normalized mutual information
    """
    input_ = kwargs.pop('input', None)
    if input_ is None:
        input_ = proc_stat(result_fn, stat_fn, **kwargs)
    kwargs.pop('cutoff')  # TODO: pop the cutoff to avoid merging again
    entropy_funcs = kwargs.pop('stat_funcs', _default_statsfuncs())
    cond_roots, cond_children, (mi, nmi) = get_entropy(input_,
            entropy_funcs, cutoff=1, **kwargs)
    return(namedtuple('MutualResult',
        'raw_counts cond_roots cond_children mi nmi')(
        input_, cond_roots, cond_children, mi, nmi))


def run_permute(merge_result, **kwargs):
    """
    conduct permutation result with levels and root labels

    Parameters
    ----------
    @param stat_func the stat_func used to compute the desired statistics
    @param num_trials is a integer number for generate how many replica
    @param stat_fn is the file name of multi-dimensional histogram stored in
    npz file format
    @param kwargs the keywords arguments needs to be passed into stat_func
    @returns a dict whose values are the statistics computed by the stat_func
    and whose keys are:
    within: permutate root label within level
    along: permute level while holding root labels unchagned
    across: permute root labels acroos different levels
    """
    assert(merge_result.ndim == 3)
    seed = kwargs.get('seed', 42)
    num_trials = kwargs.pop('num_trials', 10)
    stat_funcs = kwargs.pop('stat_funcs', _default_statsfuncs())
    rng = np.random.RandomState(seed).permutation
    max_level, n_classes, _ = merge_result.shape
    perm_indices = permutate_sampling(rng, num_trials, max_level,
            n_classes)
    tiled = np.tile(merge_result, (num_trials, 1, 1, 1))
    perm_res = {}
    for name, inds in perm_indices:
        assert(np.any(tiled != tiled[inds]))
        input_ = np.concatenate([merge_result[np.newaxis, ...], tiled[inds]])
        perm_res[name] = run_analysis(input=input_, stat_funcs=stat_funcs,
                axis_to_level=1, **kwargs)
    return(perm_res)


def plot_mutuals(samples, ax, marginal_axis, std_plot=True, **kwargs):
    """
    plot mutual information for each sentiment class

    Parameters
    ----------
    @param samples is a 4D array of dim as #reps x #level x n_classes x
    n_classes which stores the statistics result needs to be visualized
    @param ax is a matplotlib.axe object
    @param marginal_axis is the axis used to sumup in order to acquire the
    marginal probability
    @param std_plot is a boolean variable which indicates if plotting std in
    both sides @param kwargs is keywords arguments passed to plot function
    @returns the computed mean and std for replica
    """
    assert(samples.ndim == 4)
    title = kwargs.pop('title', '')
    xy_labels = kwargs.pop('xy_labels', ['', ''])
    sample_size, max_level, n_classes, _ = samples.shape
    # sum firstly to get the actual statistics
    mutual_per_level = np.mean(np.sum(samples, axis=marginal_axis), axis=0)
    sds_per_level = None
    if sample_size > 1:
        sds_per_level = np.std(np.sum(samples, axis=marginal_axis), axis=0,
                ddof=1)

    markers = ['*', 's', 'd', 'h', 'o']
    ax.set_title(title, size='medium')
    ax.set_xlim([-0.5, max_level - 0.5])
    ax.set_xticks(np.arange(0, max_level, dtype=np.int))
    ax.set_xticklabels(np.arange(1, max_level + 1, dtype=np.int))
    ax.set_xlabel(xy_labels[0], size='small')
    ax.set_ylabel(xy_labels[1], size='small')

    for i in range(n_classes):
        ax.plot(np.arange(max_level), mutual_per_level[:, i], label=i,
                linestyle='dashed', marker=markers[i], color=sent_colors[i],
                **kwargs)
        if sample_size > 1 and std_plot:
            ax.fill_between(np.arange(max_level), mutual_per_level[:, i] -
                    sds_per_level[:, i], mutual_per_level[:, i] +
                    sds_per_level[:, i], alpha=0.5)
    return(mutual_per_level, sds_per_level)


def plot_total(samples, labels, ax, legend_kwargs={}, **kwargs):
    """
    plot total mutual information

    Parameters
    ----------
    @param samples is a 4D array of dim as #reps x #level x n_classes x
    n_classes which stores the statistics result needs to be visualized
    @param labels is the labels used to annotate data
    @param ax is a matplotlib.axe object
    @param marginal_axis is the axis used to sumup in order to acquire the
    marginal probability
    @param std_plot is a boolean variable which indicates if plotting std in
    both sides @param kwargs is keywords arguments passed to plot function
    @returns the computed mean and std for replica
    """
    # plot property
    max_level = kwargs.pop('max_level', samples[0].shape[1])
    title = kwargs.pop('title', '')
    xy_labels = kwargs.pop('xy_labels', ['', ''])
    ax.set_title(title, size='medium')
    ax.set_xlim([-0.5, max_level - 0.5])
    ax.set_xticks(np.arange(0, max_level, 2, dtype=np.int))
    ax.set_xticklabels(np.arange(1, max_level + 1, 2, dtype=np.int),
            size='small')
    ax.set_xlabel(xy_labels[0], size='small')
    ax.set_ylabel(xy_labels[1], size='small')
    for i, sample in enumerate(samples):
        ttl_mutual_per_level = np.sum(np.mean(sample, axis=0), axis=(-2, -1))
        ax.plot(np.arange(max_level), ttl_mutual_per_level, linewidth=2.0,
                linestyle='-.', label=labels[i], **kwargs)
    ax.legend(**legend_kwargs)
    return(ax)


def plot_univariate_distribution(var_in_interest, stat, ax, **kwargs):
    """
    plot uni-variate distribution of samples including histogram of variable
    such as levels, roots labels and children labels
    Parameters
    ----------
    @param var_in_interest is a VARS type to indicate which axis is used for
    visualization
    @param stat is 4D numpy array
    @param ax is a matplotlib.axes object
    @kwargs is the keyword arguments passing into plot functions
    @returns ax
    """
    assert(stat.ndim == 4)
    sumup_axes = set(list(VARS)) - set([var_in_interest])
    ax.set_title('Histogram of %s' % (str(var_in_interest)), size='medium')
    ax.set_xticks(np.arange(0, stat.shape[var_in_interest], 2, dtype=np.int))
    ax.set_xticklabels(np.arange(1, stat.shape[var_in_interest] + 1, 2,
        dtype=np.int), size='small')
    bars = ax.bar(np.arange(stat.shape[var_in_interest]),
            stat.sum(axis=tuple(sumup_axes)).mean(axis=0), **kwargs)
    return(bars)


def plot_multivaraite_distribution(vars_in_interest, conditional_var, stat,
        ax=None, **kwargs):
    """
    plot joined and marginal multi-variate distribution of samples including
    histogram of joined distribution of levels, roots labels and children
    labels

    Parameters
    ----------
    @param var_in_interest is a VARS type to indicate which axis is used for
    visualization
    @param conditional_var is the variable on which joint probability of
    vars_in_interest is conditioning on; this will be computed by summing up
    the rest of axes
    @param stat is 4D numpy array storing multi-dimensional histogram result
    @param fig is a figure for plotting result
    @param kwargs is keyword argument passing into plot function
    """
    assert(stat.ndim == 4)
    joint_sumup = set(list(VARS)) - set([conditional_var] + vars_in_interest)
    cond_sumup = set(list(VARS)) - set([conditional_var])
    probs = stat.sum(axis=tuple(joint_sumup), keepdims=True) / \
            stat.sum(axis=tuple(cond_sumup), keepdims=True)
    probs = probs.mean(axis=0).squeeze()
    if ax is not None:
        ax.matshow(probs)
    return(probs)


def dummy_get(x, *args, **kwargs):
    return(x)


def plot_levels_vs_label(var_in_interest, stat, ax, **kwargs):
    probs = plot_multivaraite_distribution([var_in_interest, VARS.level],
            VARS.level, stat).squeeze()
    ttl_level, n_classes = probs.shape
    stacking = np.cumsum(np.hstack([np.zeros((ttl_level, 1)), probs[:, :4]]),
            axis=1)
    title = kwargs.pop('title', '')
    ax.set_xlim([-0.5, ttl_level - 0.5])
    ax.set_ylim([0, 1])
    ax.set_title(title, size='medium')
    ax.set_xticks(np.arange(0, ttl_level, 2))
    ax.set_xticklabels(np.arange(1, ttl_level + 1, 2), size='small')
    ax.set_xlabel('Level from Root', size='small')
    for i in range(n_classes):
        ax.bar(np.arange(ttl_level), probs[:, i], bottom=stacking[:, i],
                width=0.4, color=sent_colors[i], label=i)


def proc_length(**kwargs):
    max_level = kwargs.get('max_level', 120)
    ttl_trees = kwargs.get('ttl_trees', 11855)
    which_stat = kwargs.get('which_stat', 'level')
    binsize = kwargs.get('n_cols', 26)

    def cal_stat(stat, ngrams_dist, ncols):
        stat_dist = np.apply_along_axis(lambda arr, b:
                np.histogram(arr, b)[0], 1, stat, bins)
        stat_vs_ngrams = np.zeros((np.max(ngrams_dist) - 1, ncols))
        for stat_iter, ngram_iter in np.nditer([stat_dist, ngrams_dist],
                flags=['external_loop']):
            stat_vs_ngrams[ngram_iter - 2] += stat_iter
        return(stat_vs_ngrams)

    with open_stat(stat_fn='tree_result.npy', shape=(ttl_trees, 2, max_level),
            dtype=np.int) as tree_result:
        if which_stat == 'level':
            stat = tree_result[:, 0, :]
        else:
            stat = tree_result[:, 1, :]

        bins = np.hstack([np.arange(1, binsize + 1), max_level])
        treesize_dist = (stat > 0).sum(axis=-1) + 1
        ngrams_dist = ((treesize_dist + 1) // 2).reshape(-1, 1)
        assert(np.min(ngrams_dist) >= 2)
        stat_vs_ngrams = cal_stat(stat, ngrams_dist, binsize)
    return(stat_vs_ngrams)


def cal_boxstats(stat, cutoffs, whis):
    max_len, max_level = stat.shape
    normed_stat = stat / stat.sum(axis=0)
    acc_stat = np.cumsum(np.vstack([np.zeros((1, max_level)), normed_stat]),
            axis=0)
    length_bins = np.tile(np.arange(2, max_len + 2)[:, np.newaxis],
            (1, max_level))
    boxstats = {}
    for key, cutoff in cutoffs.items():
        boxstats[key] = length_bins[np.logical_xor((acc_stat <=
            cutoffs[key])[:-1, :], (acc_stat <= cutoffs[key])[1:, :])] + 1
    boxstats['whislo'] = boxstats['q1'] - whis * (boxstats['q3'] -
            boxstats['q1'])
    boxstats['whishi'] = boxstats['q3'] + whis * (boxstats['q3'] -
            boxstats['q1'])
    boxstats['mean'] = np.average(np.tile(2 + np.arange(stat.shape[0])[:,
        np.newaxis], (1, 26)), axis=0, weights=normed_stat)
    boxstats['fliers'] = length_bins * np.logical_or(length_bins <
            boxstats['whislo'], length_bins > boxstats['whishi'])
    boxstats['labels'] = np.arange(1, max_level + 1)
    return(boxstats)


def plot_level_properties(ax, **kwargs):
    cutoffs = kwargs.pop('cutoff', {'q1': 0.25, 'med': 0.5, 'q3': 0.75})
    whis = kwargs.pop('whis', 1.5)
    title = kwargs.pop('title', '')
    level_vs_ngrams = proc_length()
    boxstats = cal_boxstats(level_vs_ngrams, cutoffs, whis)
    fliers = boxstats.pop('fliers')
    results = pd.DataFrame(boxstats).to_dict(orient='records')
    for i in range(len(results)):
        results[i]['fliers'] = fliers[:, i][fliers[:, i] > 0]
    boxes = ax.bxp(results, **kwargs)
    ax.set_title(title, {'fontsize': 'medium'})
    ax.set_xticklabels(np.arange(1, level_vs_ngrams.shape[1] + 1, 1,
        dtype=np.int), size='small')
    return(boxes)


def plot_resample_properties(divergence, ax, **kwargs):
    colors = ['lightblue', 'lightgreen', 'darkturquoise', 'darkgreen']
    joint_name, joint_res = divergence[(VARS.root, VARS.children)][0]
    n_level = len(joint_res)
    x = np.arange(n_level, dtype=np.float)
    for i, (name, res) in enumerate(divergence[(VARS.root,)]):
        ax.bar(x, res, width=0.25, label=name, facecolor=colors[i],
                **kwargs)
        x += 0.25
    for i, (name, res) in enumerate(divergence[(VARS.children,)]):
        assert(np.allclose(res[res < 0], 0))
        res[res < 0] = 0
        ax.bar(x, res, width=0.25, label=name, facecolor=colors[i + 2],
                alpha=0.6, **kwargs)
        x += 0.25
    ax.plot(np.arange(n_level), joint_res, label=joint_name,
            linestyle='dashed', linewidth=1.5, **kwargs)
    ax.set_title('Divergence of P(root, children | level)', size='medium')
    ax.set_ylabel('Divergence', size='small')
    ax.set_xticks(np.arange(0, n_level + 1))
    ax.set_xticklabels(np.arange(1, n_level + 2), size='small')
    handlers, labels = ax.get_legend_handles_labels()
    ax.legend(handlers[1:] + [handlers[0]], labels[1:] + [labels[0]],
            fontsize='medium', fancybox=True, ncol=3, loc='upper right')


def run_stat_result(include_pattern='stat_result.*.npz', **kwargs):
    treefilename = os.path.join(data_dir, 'tree_result.npy')
    statfilename = os.path.join(data_dir, 'stat_result.npy')
    ttl_trees = kwargs.get('ttl_trees', 11855)
    batch_size = kwargs.get('batch_size', 1000)
    files = list(Path(data_dir).glob(include_pattern))
    if len(files) == 0:
        # re-run result
        handler = calstat_handler
    else:
        # load result
        assert(ttl_trees // batch_size == len(files))
        handler = loadstat_handler
    offset_, size_count, stat_result, tree_result = 0, 0, None, None
    run_batch(handler, files, None, **kwargs)
    del stat_result
    del tree_result
    return(statfilename, treefilename)


def run_boostrap_result(**kwargs):
    # read stat_result and tree_result from memmap
    treefilename = kwargs.get('treefilename', None)
    statfilename = kwargs.get('statfilename', None)
    max_level = kwargs.get('max_level', 120)
    n_classes = kwargs.get('n_classes', 5)
    ttl_trees = kwargs.get('ttl_trees', 11855)
    num_trials = kwargs.get('num_trials', 10)
    sample_scheme = kwargs.get('sample_scheme', 'conditional')
    output_file = kwargs.get('outputfilename', 'kl_divergence.npz')

    if treefilename is None:
        treefilename = os.path.join(data_dir, 'tree_result.npy')
    if statfilename is None:
        statfilename = os.path.join(data_dir, 'stat_result.npy')

    stat_result = np.memmap(statfilename, mode='r', dtype=np.float,
            shape=(max_level, n_classes, n_classes))
    tree_result = np.memmap(treefilename, mode='r', dtype=np.int,
            shape=(ttl_trees, 2, max_level))
    levels = tree_result[:, 0, :].view(np.ma.MaskedArray)
    levels.mask = levels == 0
    labels = tree_result[:, 1, :].view(np.ma.MaskedArray)
    labels.mask = np.hstack(
            [np.zeros((len(labels), 1), dtype=np.bool), levels[:, 1:] == 0])
    resamples = bootstrap_sampling((stat_result, levels, labels),
            rng.multinomial, num_trials=num_trials, priors=sample_scheme)
    result_file = os.path.join(data_dir, output_file)
    with open(result_file, 'wb') as fp:
        np.savez(fp, resamples=resamples)
        logging.info('%d KL-divergenc resample result stored in %s' %
            (num_trials, result_file))
    del stat_result
    del tree_result


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s [%(levelname)s]:%(message)s', datefmt="%Y-%m-%d %H:%S",
            level=logging.INFO)
    kwargs = dict([('ttl_trees', 11855), ('batch_size', 1000),
                   ('max_level', 120), ('n_classes', 5)])
    # run_stat_result(max_level=120, n_classes=5)
    run_boostrap_result(**kwargs)
