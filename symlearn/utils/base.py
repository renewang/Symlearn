from collections import Counter, UserDict
from contextlib import contextmanager
from functools import partial, singledispatch

import numpy
import pandas

import re
import os
import logging
import shelve
import csv

__path__= ['.', '../scripts/run_shelter_comp']

logger = logging.Logger(__name__)
data_dir = os.path.join(os.getenv('DATADIR', default='..'), 'data')

NUM_PAT = re.compile(r"([()+\-.,]?\d+)+")
PUNKT_PAT = re.compile(r"^([^\w]+)$")


class VocabularyDict(UserDict):

    def __init__(self, dict_file, counter=None, max_features=numpy.Inf,
            norm_number=None, norm_punkt=None):
        self.max_features = max_features 
        if type(dict_file) is str:  # open dictionary file via shelve
            if counter:
                func = partial(preprocess_dictionary, counter, dict_file)
            else:
                func = partial(shelve.open, dict_file, flag='r')
            with func() as shelf:
                super(VocabularyDict, self).__init__(shelf)
        elif isinstance(dict_file, (VocabularyDict, dict)): # copy constructor
            super(VocabularyDict, self).__init__(dict_file)

        if not '-unk-' in self.data:
            self.data['-unk-'] = -1  # reference to the last column in embedding
            
        if norm_number:
            self.norm_number = NUM_PAT
        else: 
            self.norm_number = None
        if norm_punkt:
            self.norm_punkt = PUNKT_PAT
        else:
            self.norm_punkt = None 

    def __missing__(self, word):
        if self.norm_number and self.norm_number.match(word):
            return(self.data['-num-'])
        if self.norm_punkt and (self.norm_punkt.match(word) or word in
                ['-lrb-', '-rrb-']):
            return(self.data['-punkt-'])
        return self.data['-unk-']

    def __setitem__(self, word, val):
        if word == '-unk-':
            val = -1
        if val < self.max_features:
            super(VocabularyDict, self).__setitem__(word, val)

    def __getitem__(self, word):
        val = super(VocabularyDict, self).__getitem__(word)
        if val == -1:
            return int(numpy.min([len(self) - 1, self.max_features]))
        else:
            return int(val)


@contextmanager
def preprocess_dictionary(counter, dictionary_file, max_features=None):
    assert(isinstance(counter, Counter))
    with shelve.open(dictionary_file, flag='n') as shelf:
        for i, (w, _) in enumerate(counter.most_common(n=max_features)):
            shelf[w] = i
        if not max_features:
            max_features = len(counter)
        logger.info("creating dictionary file %s" % dictionary_file)
        yield shelf


def compute_inverse(vocab):
    index2vocab = numpy.empty(len(vocab), dtype=numpy.object)
    index2vocab[tuple(vocab.values()), ] = list(vocab.keys())
    return(index2vocab)


def run_batch(handler, batch_size, ttl_trees, *args, **kwargs):
    """

    """
    # these two import statement here for the circular reference purpose
    # should be removed once done correctly
    from symlearn.csgraph.adjmatrix import to_csgraph
    import symlearn.csgraph.cnltk_trees as cnltk
    tree_file = kwargs.pop('tree_file', None)
    stat_files = kwargs.pop('stat_files', None)
    n_batch = numpy.ceil(ttl_trees / batch_size).astype(numpy.int)

    with shelve.open(os.path.join(data_dir, 'vocab'), flag='r') as vocab:
    # not sure what the flag is 'n' => always create a new one    
    # with shelve.open(os.path.join(data_dir, 'vocab'), flag='n') as vocab:
        index2vocab = compute_inverse(vocab)
        for i in range(n_batch):
            logger.info('%d th batch round are started' % (i))
            if tree_file:
                data = cnltk.read_bin_trees(filename,
                        list(range(i * batch_size, (i + 1) * batch_size)),
                        converted=False)
                data = [to_csgraph(d, vocab) for d in data]
                logger.info('%d th batch of trees are generated' % (i))
            else:
                assert(stat_files)
                data = stat_files[i]
            handler(i, data, index2vocab, **kwargs)
            #handler(i, data, *handler_args, **kwargs)
            logger.info('%d th batch of stat_result are stored' % (i))


@contextmanager
def check_treebased_phrases(csv_file, n_rows=None, memory_map=False):
    """
    ensure the phrases are all included if the sentence id is also included in
    data set

    Parameters
    ==========
    @param csv_file: string type
        used to pass to pandas read_table
    @param n_rows: int or None 
        used to indicate how many rows should be read in. If None, all the
        records will be read in
    @param memory_map: boolean
        used to indicate if usin memory map with pandas.read_csv  
    """
    cur_dialect = csv.unix_dialect()
    cur_dialect.skipinitialspace=True
    dtypes_ = {'tree_id': numpy.int16, 'node_id': numpy.int16, 'sentiment': numpy.int8,
            'level': numpy.int16, 'start_pos': numpy.int32, 'end_pos': numpy.int32}
    phrases = pandas.read_table(csv_file, delimiter=',', header=0, dialect=cur_dialect,
            nrows=n_rows, memory_map=memory_map, dtype= dtypes_,
            index_col=False)  # adding index_col=False to disable using first
                              # column as index
    treeid_grps =  phrases.groupby(phrases['tree_id'].values)
    ids, tree_sizes = map(numpy.asarray, zip(*[(gid, len(grp) - 1) for gid, grp in
       treeid_grps]))
    
    if n_rows: # only check if n_rows is not None
        # 1. checking the root has node_id equals 0
        sentences = treeid_grps.nth(0)
        assert(all(sentences['node_id'] == 0))
        
        # 2. excluding those truncating records due to not all of n_rows read in
        test_phrases = phrases[phrases['node_id'] != 0]

        ids = ids[tree_sizes!=0]
        phrases = pandas.concat([treeid_grps.get_group(i) for i in ids])
        assert(len(test_phrases)== numpy.sum(tree_sizes))
        tree_sizes = tree_sizes[tree_sizes != 0]

        # 3. checking the start_pos and end_pos will map to the same extracted phrases
        sentences_toks = sentences['phrase'].apply(lambda x: x.split()) 
        assert(all(test_phrases.apply(
            lambda rec, toks: ' '.join(
                toks[rec['tree_id']][rec['start_pos']:rec['end_pos']]) == rec['phrase'],
            axis=1, broadcast=False, raw=False, reduce=False,
            args=(sentences_toks,))))

    yield phrases, tree_sizes


def compute_len_stats(len_dist):
    """
    return lengths distribution of sentence plus cumulative sum
    @param len_dist is a list whose elements are corresponding sentence length
    """
    # TODO: consider to replace with numpy.bincount
    len_count = numpy.zeros(len(numpy.unique(len_dist)) + 1, dtype=numpy.int)
    for i, cur_len in enumerate(numpy.unique(len_dist)):
        len_count[i + 1] = numpy.sum(len_dist == cur_len)
    return(numpy.unique(len_dist), len_count)


@singledispatch
def get_phrases_helper(first, trees, vocab=None):
    raise ValueError("fails to find the corresponding type {!r}".format(
        first.__class__))


def fit_transform(inst, X, y=None, **fit_param):
    """
    current patch to solve FunctionTransformer problem
    """
    from . import WordIndexer
    if isinstance(inst.func, WordIndexer):
        return(inst.transform(X, y, **fit_param))
    elif fit_param.get('is_order', None):
        return(inst.transform(X, y))
    else: # identity transform
        if y is not None:
            yt = [yy.reshape(-1, 1) if yy.ndim == 1 else yy for yy in y]
        else:
            yt = y
        return(X, yt)