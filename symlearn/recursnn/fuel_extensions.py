from operator import itemgetter

from symlearn.fuel.transformers import Transformer
from symlearn.fuel.transformers import Batch
from symlearn.fuel.transformers import SourcewiseTransformer
from symlearn.fuel.datasets import IndexableDataset
from symlearn.fuel.streams import DataStream
from symlearn.fuel.schemes import BatchScheme
from symlearn.fuel.schemes import IndexScheme
from symlearn.fuel.schemes import ConstantScheme

from collections import Iterable
from picklable_itertools import iter_
from itertools import islice
from itertools import zip_longest
from itertools import tee
from itertools import chain
from itertools import groupby

from . import recursnn_helper
from . import recursnn_utils
from . import recursnn_train

import os
import pickle
import numpy as np
import scipy as sp
import logging

logger = logging.getLogger(__name__)

class Stacking(SourcewiseTransformer):

    def __init__(self, data_stream, which_sources=None,
            **kwargs):
        super(__class__, self).__init__(data_stream,
                data_stream.produces_examples, which_sources=which_sources)

    def _helper(self, data, *args):
        if isinstance(data, (list, tuple)):
            if isinstance(data[0], sp.sparse.spmatrix):
                return sp.sparse.vstack(data)
            elif isinstance(data[0], np.ndarray):
                return np.vstack(data)

    def transform_source_example(self, source_example, _):
        return(self._helper(source_example))

    def transform_source_batch(self, source_batch, _):
        return(self._helper(source_batch))

class SourcewiseMapping(SourcewiseTransformer):

    def __init__(self, data_stream, mapping, which_sources=None,
            **kwargs):
        self.mapping = mapping
        super(__class__, self).__init__(data_stream,
                data_stream.produces_examples, which_sources=which_sources)

    def transform_source_example(self, source_example, source_name):
        return self.mapping(source_example)

    def transform_source_batch(self, source_batch, source_name):
        return self.mapping(source_batch)

def get_minibatch(test_size, dataset=None, indices=None, num_fold=1,
                  random_state=42):
    """
    conduct an initial training and test split according to sentence length

    @param test_size is the proportion of test_size will be used for test in
                     float
    @param dataset is the dataset will be used for split, usually is the
                    lengths of sentences
    @param indices is the corresponding mapping indices for dataset
    @param num_fold is the number of iterations will be conducted in shuffle
                    split (currently can only be one)
    @param random_state is the seed for generating random split
    """
    if dataset is None:
        for varname in ['full_sents', 'trees', 'forests']:
            # load from files
            if not (varname in locals() or varname in globals()):
                with open(os.path.join(os.environ['WORKSPACE'],
                                       'Kaggle/skeleton/data/' +
                                       varname + '.pickle'), 'rb') as fp:
                    if varname == 'full_sents':
                        full_sents = pickle.load(fp)
                    elif varname == 'trees':
                        logger.info("TOFIX: currently loading from mmap file")
                    elif varname == 'forests':
                        forests = pickle.load(fp)
        dataset = full_sents['Length'].values
        indices = np.asarray([forests[idx] - 1 for idx in full_sents.index])

    if test_size > 0.0 and test_size < 1.0:
        cv = recursnn_train._setup_cv(dataset, nfold=num_fold,
                                      random_state=random_state,
                                      test_size=test_size,
                                      foldname='shufflesplit')
    else:
        cv = zip_longest([(np.arange(len(dataset) * int(1 - test_size)))],
                [(np.arange(len(dataset) * int(test_size)))])

    if indices is None:
        indices = np.arange(len(dataset))

    if not isinstance(indices, np.ndarray):
        indices = np.asarray(indices)

    if not isinstance(dataset, np.ndarray):
        dataset = np.asarray(dataset)

    for train, test in cv:
        yield (indices[train], dataset[train], indices[test], dataset[test])


def iter_minibatches(batch_iter, minibatch_size, dry_run=False):
    """
    >>> np.all(list(iter_minibatches((np.arange(10), np.arange(11, 1,
    ... -1), [], []), 1, True))[0][0] == np.arange(9, -1, -1))
    True

    >>> np.all(list(map(itemgetter(0), list(iter_minibatches((np.arange(20),
    ... np.arange(20)//4, [], []), 1, True)))) == np.arange(20).reshape(4, 5,
    ... order='F'))
    True

    >>> np.all(list(map(itemgetter(0), list(iter_minibatches((np.arange(20),
    ... np.arange(20)//4, [], []), 2, True)))) ==
    ... np.sort(np.arange(20).reshape((2, 2, 5),
    ... order='F').swapaxes(0,1).reshape(2, 10)))
    True

    >>> np.all(list(map(itemgetter(0), list(iter_minibatches((np.arange(20),
    ... np.arange(20)//4, [], []), 3, True)))) == np.asarray([[0, 1, 2, 4, 5,
    ... 6, 8, 9, 10, 12, 13, 14, 16, 17, 18], [3, 7, 11, 15, 19]]))
    True

    >>> np.all(list(map(itemgetter(0), list(iter_minibatches((np.arange(20),
    ... np.arange(20)//4, [], []), 4, True)))) == np.arange(20))
    True
    """

    tree_indices, tree_sizes, _, _ = batch_iter
    # pre-sort training index by tree_size
    presorted = [(size, index)
                 for index, size in zip(tree_indices, tree_sizes)]
    presorted.sort(key=itemgetter(0))

    def pairwise(iterable):
        a, b = tee(iterable)
        next(b, None)
        return(zip(a, b))

    break_points = list(map(lambda x: itemgetter(0)(next(x)),
                        map(itemgetter(1), groupby(
                            [(idx, size) for idx, (size, _)
                             in enumerate(presorted)], key=itemgetter(1)))))
    break_points.append(len(presorted))
    assert(len(break_points) == len(np.unique(tree_sizes)) + 1)
    len2bpts = {size: bp for bp, size in zip(
        break_points[1:], np.unique(list(map(itemgetter(0), presorted))))}

    if minibatch_size < 1:
        prop_minibatch = minibatch_size
    else:
        prop_minibatch = minibatch_size * len(len2bpts) / len(presorted)

    minibatch_size = int((len(presorted) * prop_minibatch) // len(len2bpts)) \
            or 1

    logger.info('using batch size = {} for provided proportion {:.2f} '
                'for each length group'.format(minibatch_size, prop_minibatch))
    cur_pointers = [islice(range(len(presorted)), start, stop, minibatch_size)
                    for start, stop in pairwise(break_points)]

    training_inds = []
    for presorted_iter in zip_longest(*cur_pointers):
        training_inds.append([])
        for start in presorted_iter:
            if start is None:
                continue
            end = start + minibatch_size
            if end >= len2bpts[presorted[start][0]]:
                end = len2bpts[presorted[start][0]]
            training_inds[-1].extend(list(chain(map(itemgetter(1),
                presorted[start:end]))))
    logger.info(
        'total number of iterations is about {}'.format(len(training_inds)))
    for batch_inds in training_inds:
        if dry_run:
            undertest = None
        else:
            training_set = read_bin_trees(os.path.join(
                os.environ['WORKSPACE'],
                'Kaggle/skeleton/data/trees.bin'),
                [index for index in batch_inds])
            undertest = [(len(tree.leaves()), tree) for tree in training_set]
        yield batch_inds, undertest


class ExternalIndexScheme(IndexScheme):
    """
    given an external reference such as pre-split train/test set or
    preprocessing such as grouping or sorting
    """

    def __init__(self, index_ref):
        super(__class__, self).__init__(index_ref)

    def get_request_iterator(self):
        return iter_(self.indices)


class GroupIndexScheme(ExternalIndexScheme):
    """
    given group_keys iterate sample based on the group result
    """
    def __init__(self, group_keys, index_ref=None):
        if index_ref is None:
            index_ref = len(group_keys)
        self.group_keys = group_keys
        super(__class__, self).__init__(index_ref)

    def get_request_iterator(self):
        output = itemgetter(0)(list(zip(*iter_minibatches(
            (self.indices, self.group_keys, [], []), 1,
            dry_run=True))))
        return iter_(output)


class ExtractFeatures(object):

    def __init__(self, **kwargs):
        self.params = kwargs

    def __call__(self, batch):
        if not sp.sparse.issparse(*batch):
            est_gtree, est_cost = \
                recursnn_utils.build_greedy_tree(*batch, **self.params)
            batch = recursnn_utils.tree_to_matrix(est_gtree)
        examples = recursnn_helper._iter_matrix_groups(batch, **self.params)
        if hasattr(self, 'successor'):
            return(self.successor(batch+examples))
        else:
            return(examples)

    def setSuccessor(self, successor):
        self.successor = successor


class SplitSources(Transformer):
    """
    split selected sources into desired sources
    """
    def __init__(self, data_stream, mapping, add_sources, **kwargs):
        self.mapping = mapping
        self.add_sources = add_sources
        super(__class__, self).__init__(data_stream, **kwargs)

    @property
    def sources(self):
        split_sources = []
        for source in self.add_sources:
            if source:
                split_sources += tuple([source])
        return self.data_stream.sources + tuple(split_sources)

    def get_data(self, request=None):
        if request is not None:
            raise ValueError

        data = next(self.child_epoch_iterator)
        split = self.mapping(data)
        assert(len(split) == len(self.add_sources))
        for sub in zip(split):
            data += sub
        return(data)


class SlidingWindowScheme(IndexScheme):

    def __init__(self, windowsize, examples, fill_value=-1):
        self.windowsize = windowsize
        self.constant_value = fill_value
        super(__class__, self).__init__(examples)

    def get_request_iterator(self):
        lpadded = np.lib.pad(self.indices, (self.windowsize//2,)*2, 'constant',
                             constant_values=self.constant_value)
        out = [lpadded[i:i+self.windowsize] for i in range(len(self.indices))]
        return(iter_(out))


class StreamWrapper(Transformer):
    """
    a simple wrapper to turn elements in one IterableDataset into
    IndexableDataset with SlidingWindowScheme
    """
    def __init__(self, datastream, window_size, **kwargs):
        self.window_size = window_size
        self.curtime_stream = None
        super(__class__, self).__init__(datastream, **kwargs)

    def get_data(self, request=None):
        if request is not None:
            raise ValueError
        assert(self.curtime_stream is None)
        data = next(self.child_epoch_iterator)
        time_steps = len(*data) - 1
        assert(isinstance(np.array(*data), Iterable))
        self.curtime_stream = Batch(DataStream(
            IndexableDataset(*data), iteration_scheme=SlidingWindowScheme(
                self.window_size, time_steps)),
            iteration_scheme=ConstantScheme(time_steps))
        time_slices = [timestep for timestep in
                       self.curtime_stream.get_epoch_iterator()]
        assert(len(time_slices) == 1)
        self.curtime_stream = None  # nullify used curtime_stream
        return(time_slices.pop())
