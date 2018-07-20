from functools import singledispatch
from functools import partial
from itertools import repeat
from operator import itemgetter
import logging
import numpy

logger = logging.getLogger(__name__)


@singledispatch
def sample_scheme(sample_sizes, labels, rng, priors):
    """
    Function for generate multi-nomial distribution in different prior schemes

    Parameters
    ----------
    @param sample_sizes is dispatching parameter should store the
    sample sizes used for multinomial distribution
    @param labels is an array storing the labels distribution used for
    computing prior distribution
    @param rng is a numpy random generator partially bound number of trials
    @returns the generating replica
    """
    raise(ValueError('not recognizable sample_sizes {:s}'.format(
        str(type(sample_sizes)))))


@sample_scheme.register(int)
def sample_from_total(sample_sizes, labels, rng, priors):
    """
    Overloading function for sample_scheme with the first argument as integer

    Parameters
    ----------
    @param sample_sizes is total number of sample
    @param labels is an array storing the labels distribution used
    for computing prior distribution or an 1D vector which has lenght equals to
    number of labels to store manually given priors
    @param rng is a numpy random generator partially bound number of trials
    @returns the generating replica
    """
    n_classes = len(set(labels.compressed()))
    roots, root_counts = numpy.unique(labels[:, 0],
            return_counts=True)
    if len(roots) < n_classes:
        logger.warn('root is undersampling for labels: {!r}'.
                format(set(range(n_classes)) - set(roots)))
    # compute from children
    probs = compute_priors(labels, priors, n_classes)
    root_dist = numpy.zeros(n_classes, dtype=numpy.int)
    root_dist[roots] = root_counts
    adj = (root_dist / root_dist.sum()).reshape(1, n_classes, 1)
    replica = numpy.atleast_3d(rng(sample_sizes, probs)).swapaxes(1, -1)
    assert(numpy.allclose((adj * replica).sum(axis=1, keepdims=True), replica))
    replica = adj * replica
    return(replica)


@sample_scheme.register(numpy.ndarray)
def sample_from_conditional(sample_sizes, labels, rng, priors):
    """
    Overloading function for sample_scheme with the first argument as numpy
    array to specify the sample_sizes based on different priors

    Parameters
    ----------
    @param sample_sizes is an numpy array where the row are the conditional
    variable (could be the root label distribution or children nodes
    distribution) and the columns are the labels distributions based on which
    random multinomial samples will be generating
    @param labels is an array storing the labels distribution used for
    computing prior distribution
    @param rng is a numpy random generator partially bound number of trials
    @returns the generating replica
    """
    n_classes = sample_sizes.shape[0]
    n_trials = rng.keywords['size']
    probs = compute_priors(labels, priors, len(sample_sizes))
    replica = numpy.empty((n_trials, n_classes, n_classes),
            dtype=sample_sizes.dtype)
    for i in range(len(sample_sizes)):
        replica[:, i, :] = rng(sample_sizes[i], probs[i])
    return(replica)


def compute_priors(labels, priors, n_classes=5):
    """
    Function to compute empirical prior distribution from data

    Parameters
    ----------
    @param labels is a collection of labels data from sample and is used to
    compute priors
    @param priors is a string and used to tell function how to compute priors
    @param n_classes is the number of total labels
    @returns the computed priors based on the priors computation scheme
    """
    if type(priors) != str:  # return given
        probs = priors
        if not numpy.isclose(probs.sum(), 1):  # if not normalized
            probs /= probs.sum()
        return(probs)
    roots = numpy.asarray([numpy.sum(labels[:, 0] == i) for i in
        range(n_classes)])
    # joint probability
    probs = numpy.asarray([numpy.sum(numpy.logical_and(
        (labels[:, 0] == i)[:, numpy.newaxis], labels[:, 1:] == j))
        for i in range(n_classes) for j in range(n_classes)]).reshape(
                n_classes, n_classes)
    if priors == "conditional":
        # conditional node labels distribution given roots labels
        probs = numpy.nan_to_num(probs / roots[:, numpy.newaxis])
        denominator = numpy.sum(probs, axis=1, keepdims=True)
    elif priors == "total":
        # total node labels distribution discounting root labels
        probs = numpy.sum(probs, axis=0)
        denominator = numpy.sum(probs)
    else:
        raise ValueError('no such {:s} sample scheme'.format(priors))
    probs = numpy.nan_to_num(probs / denominator)
    return(probs)


def bootstrap_sampling(tree_stats, rng, num_trials=100, priors="conditional"):
    """
    Public and convenient function to be called in order to generate bootstrap
    resampling. Caller is responsible to pass the precomputed histograms and
    sample results and store in tree_stats parameter. This mainly including:

    stat_result: a multi-dimensional histograms where store the count divided
    by the levels, root node labels and children node labels. The first
    dimension of histogram is level, the second is the root label and the last
    is the children node label

    levels: a padded list to store the level value for each sample. Each sample
    will be presented as a sub-list and padded with zeros at the end in order
    to have the same length with the sample of maximal length. The first column
    is usually the root and its level should be zero.

    labels: a padded list to store the level value for each sample. Each sample
    will be presented as a sub-list and padded with zeros at the end in order
    to have the same length with the sample of maximal length. The first column
    is usually the root.

    The resampling will based on the scheme specified in the probs parameter.
    Currently, supporting two scheme: one and default is 'conditional' which
    will compute prior distribution of children node labels conditional on the
    root labels; the other is 'total' which will compute prior distribution of
    children node labels from all samples disregarding the root label

    Parameters
    ----------
    @param tree_stats is a collection of recomputing statstics sample result,
    @param rng is a numpy random generator will be used to generate samples
    @param num_tirals is the number of trials to generate replica
    @param priors is either a string used to specify how to compute the label
    priors from sample or a 1D numpy array as given prior
    @returns the computed priors based on the priors computation scheme
    """
    bound_rng = partial(rng, size=num_trials)
    stat_result, levels, labels = tree_stats
    max_level, n_classes = stat_result.shape[:2]
    if (isinstance(priors, (list, numpy.ndarray)) and
            len(priors) == n_classes) or priors == "total":
        # adding asscalar to enforce as int type
        sample_sizes = numpy.asscalar(stat_result.sum().astype(numpy.int))
    elif priors == "conditional":
        sample_sizes = stat_result.sum(axis=(0, -1)).astype(numpy.int)
    else:
        raise ValueError('no such {:s} sample scheme'.format(priors))
    replica = sample_scheme(sample_sizes, labels, bound_rng, priors)
    resamples = _place_replica(stat_result, replica)
    assert(resamples.sum() == num_trials * stat_result.sum())
    assert(numpy.allclose(resamples.sum(axis=(0, -2, -1)) / resamples.sum(),
        stat_result.sum(axis=(1, -1)) / stat_result.sum(), atol=1e-4))
    return(resamples)


def _place_replica(stat_result, replica):
    """
    A helper function in order to recompute the multi-dimensional histogram
    (level x root labels x children labels) based on the random genreating
    resample

    Parameters
    ----------
    @param stat_result is the precomputed multidi-mensional histogram from real
    samples
    @param replica is the generated random sample based on specified sample
    scheme
    @return the historgram computed based on the resample
    """
    # replica needs to be in # trials x 1 x # roots x # nodes
    replica = replica.reshape((replica.shape[0], 1) + replica.shape[1:])
    prop_by_level = numpy.nan_to_num(stat_result.sum(axis=-1) /
            stat_result.sum(axis=(0, -1)))
    # prop_by_level needs to be in 1 x # levels x # roots x 1
    prop_by_level = prop_by_level.reshape((1, ) + prop_by_level.shape + (1, ))
    resamples = prop_by_level * replica
    assert(numpy.allclose(resamples.sum(axis=1), replica[:, 0, :, :]))
    return(resamples)


def load_and_modify(memmap_filename, update_res, handler, memmap_fp=None,
        **kwargs):
    memmap_fp = handler(update_res, memmap_fp, memmap_filename, **kwargs)
    iter_ = numpy.nditer([update_res, memmap_fp],
                flags=['external_loop', 'buffered'],
                op_flags=[['readonly'], ['writeonly', 'allocate']])
    for from_arr, to_arr in iter_:
        to_arr[:len(from_arr)] += from_arr
    return(memmap_fp)


def _permute_within_level(rng, rep_size, max_level, n_classes):
    # permute parent labels
    # using stat_result[numpy.arange(max_level)[:, numpy.newaxis],
    # permute_within_level,:] to access permuated reps
    permute_ = numpy.asarray(list(map(rng,
        repeat(numpy.arange(n_classes), rep_size * max_level)))).reshape(
                rep_size, max_level, n_classes).astype(numpy.intp)
    assert(numpy.all(itemgetter(1)(numpy.unique(permute_,
        return_counts=True)) == numpy.repeat(rep_size * max_level, n_classes)))
    rep_inds, level_inds, _ = numpy.indices((rep_size, max_level, n_classes))
    permute_within_level = (rep_inds, level_inds, permute_)
    return(permute_within_level)


def _permute_along_level(rng, rep_size, max_level, n_classes):
    # using stat_result[permute_along_level]
    permute_ = numpy.asarray(list(map(rng, repeat(numpy.arange(max_level),
        rep_size)))).reshape(rep_size, max_level).astype(numpy.intp)
    rep_inds, _ = numpy.indices((rep_size, max_level))
    permute_along_level = (rep_inds, permute_)
    return(permute_along_level)


def _permute_across_level(rng, rep_size, max_level, n_classes):
    # using stat_result[permute_across_level[0][:,:,numpy.newaxis],
    # permute_across_level[1], :]
    permute_ = numpy.asarray(list(map(rng, repeat(numpy.arange(
        max_level * n_classes), rep_size)))).reshape(
            rep_size, max_level, n_classes).astype(numpy.intp)
    rep_inds, _, _ = numpy.indices((rep_size, max_level, n_classes))
    permute_across_level = (rep_inds, permute_ // n_classes,
            permute_ % n_classes)
    return(permute_across_level)


def permutate_sampling(rng, rep_size, max_level, n_classes, schemes_funcs=[
        ('within', _permute_within_level), ('along', _permute_along_level),
        ('across', _permute_across_level)]):
    perm_reps = [(name, func(rng, rep_size, max_level, n_classes)) for name,
            func in schemes_funcs]
    return(perm_reps)
