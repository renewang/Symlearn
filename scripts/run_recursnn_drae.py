from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import TransformerMixin
from sklearn.linear_model.base import LinearClassifierMixin
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import (ShuffleSplit, StratifiedShuffleSplit,
                                      KFold)
from sklearn.preprocessing import FunctionTransformer
from sklearn.utils.validation import check_is_fitted, NotFittedError
from nltk.tokenize import SpaceTokenizer

from symlearn.utils import (WordIndexer, fit_transform, VocabularyDict)
from symlearn.recursnn.recursive import RecursiveBrick
from symlearn.recursnn.recursnn_rae import RecursiveAutoEncoder
from symlearn.recursnn.recursnn_drae import RecursiveTreeClassifier, schedule_learning
from symlearn.recursnn.recursnn_helper import _iter_matrix_groups
from exec_helper import patch_pickled_preproc

from operator import itemgetter
from unittest.mock import patch
from contextlib import contextmanager
from itertools import accumulate

import memory_profiler as mprof

import h5py
import os
import theano
import numpy
import logging
import inspect
import numbers
import types
import joblib
import operator

FORMAT='%(asctime)s:%(levelname)s:%(threadName)s:%(filename)s:%(lineno)d:%(funcName)s:%(message)s'
logging.basicConfig(format=FORMAT, datefmt='%m/%d/%Y %I:%M:%S %p',
        level=logging.INFO)

### for memory_profile logging
# create file handler
fh = logging.FileHandler("memory_profile.log")
fh.setLevel(logging.DEBUG)

# create formatter
formatter = logging.Formatter('%(message)s')
fh.setFormatter(formatter)

memprof_logger = mprof.LogFile('memory_profile_log', reportIncrementFlag=True)
memprof_logger.logger.addHandler(fh)
####


@contextmanager
def proc_data(filename):
  basename = os.path.splitext(filename)[0]
  with open(filename, 'r+b') as fp:
    try:
      root = h5py.File("%s.hdf5" % (basename), mode='w-')
      vstr_dtype = h5py.special_dtype(vlen=str)
      tree_strs = fp.readlines()
      raw_data = root.create_dataset("tree_strs", shape=(len(tree_strs),), dtype=vstr_dtype)
      raw_data[:] = tree_strs
      root.flush()
      del tree_strs
    except OSError:
      root = h5py.File("%s.hdf5" % (basename), mode='r')
      raw_data = root['tree_strs']
  yield(raw_data)
  root.close()


def incremental_learning(classifier, train_data, valid_data, train_sizes, verbose=1,
        params={}):
    n_trainsizes  = (len(train) * train_sizes).astype(numpy.int)
    train_scores, valid_scores = \
            numpy.zeros((len(n_trainsizes), 2)), numpy.empty((len(n_trainsizes), 2))
    train_features, train_targets = train_data 
    train_roots = numpy.asarray([t[0] for t in train_targets]).ravel()
    valid_features, valid_targets = valid_data 
    valid_roots = numpy.asarray([t[0] for t in valid_targets]).ravel()
    classes_ = numpy.unique(valid_roots)

    for i, start in enumerate([0] + n_trainsizes[:-1].tolist()):
        end = n_trainsizes[i]
        logging.info("start {:d}-th training, increment sample size to {:d}, "
                     "distribution of training labels {}".format(i + 1, end,
                         numpy.bincount(train_roots)))
        assert(all(numpy.diff([f.nnz for f in train_features[start:end]]))>=0)
        classifier.autoencoder = classifier.autoencoder.fit(
                train_features, **fit_params)
        classifier.fit(train_features[start:end], train_targets[start:end],
                **params)
        train_scores[i] = classifier.score(train_features[start:end],
                train_targets[start:end], 'error_rate')
        valid_scores[i] = classifier.score(valid_features, valid_targets,
                'error_rate')
        logging.info("complete {:<d}th training, valid size = {:<d}, "
                "distribution of " "labels {}, train_score={}, "
                "valid_score={}".format(i + 1, len(valid),
                    numpy.bincount(valid_roots),
                    numpy.array_str(numpy.asarray(train_scores[i]), precision=4,
                        suppress_small=True),
                    numpy.array_str(numpy.asarray(valid_scores[i]), precision=4,
                        suppress_small=True)))
        if i == 0:
            break
    return(n_trainsizes, train_scores, valid_scores)

 

def load_preproc(**kwargs):
    pretrain_loc = kwargs.get('pretrain_loc', None)
    preproc = patch_pickled_preproc(pretrain_loc)
    # hacks to shrink the vocabulary size along with components_
    # only keep 100 for testing, should be removed for the final model
    ovocab = preproc.named_steps['vectorizer'].func.vocab
    # manually patch max_features default value
    ovocab.max_features = numpy.inf
    nvocab = VocabularyDict(ovocab, max_features=100)
    preproc.named_steps['vectorizer'].func.vocab = nvocab
    oembed = preproc.named_steps['decomposer'].components_
    nembed = numpy.empty((oembed.shape[0], 101), dtype=oembed.dtype)
    nembed[:, :100]= oembed[:, :100]
    nembed[:, -1] = oembed[:, -1]
    preproc.named_steps['decomposer'].components_ = nembed
    preproc = Pipeline(steps=[
        ('indexer', FunctionTransformer(func=WordIndexer(preproc),
            validate=False, pass_y=True)),
        ('sorter', FunctionTransformer(func=schedule_learning(),
            validate=False, pass_y=True)),
    ])
    for name, trans in preproc.steps:
        setattr(trans, 'fit_transform', types.MethodType(fit_transform, trans))
    return(preproc)


def construct_model(**kwargs):
    classifier = RecursiveTreeClassifier(**kwargs)
    return(classifier)


def preprocess(preproc, raw_data, **fit_params):
    assert(len(raw_data) <= 2)
    if len(raw_data) == 1:  # training data is parsing tree
        raw_x = raw_data[0]
        raw_y = [None for i in range(len(raw_x))] 
    else:
        raw_x, raw_y = raw_data
    if not isinstance(raw_x, (numpy.ndarray)):
        features = numpy.asarray(raw_x)
    else:
        features = raw_x
    if not isinstance(raw_y, (numpy.ndarray)):
        targets = numpy.asarray(raw_y)
    else:
        targets = raw_y
    features, targets = preproc.fit_transform(features, targets,
            **fit_params)
    assert(not fit_params.get('sorter__is_order') or all(numpy.diff([f.nnz for
        f in features])>=0))
    return(features, targets)


def proc_learning(classifier, train_data, valid_data, **fit_kws):
    samplesizes = fit_kws.get("train_sizes", numpy.linspace(0.1, 1.0, 5))
    learning_func = fit_kws.pop("learn_func", incremental_learning)
    scorer = fit_kws.pop('scorer', None)
    fit_params = fit_kws.pop('fit_params')

    ttl_size = numpy.asarray(train_features).size * \
        numpy.asarray(train_features).dtype.itemsize * \
            numpy.max(samplesizes) / (1024**2)

    logging.info('start training with training size %d (%.2f MB in memory)' % (
            len(train_features) * numpy.max(samplesizes), ttl_size))
    start_time = time.time()
    learning_res = learning_func(classifier, train_data, valid_data,
            train_sizes=samplesizes, verbose=1, params=fit_params)

    logging.info('complete training in total time %.2f seconds' % (time.time() - start_time))
    logging.info('sample size is {!r}'.format(learning_res[0]))

    logging.info('training score of root labels is {!r}'.format(
        numpy.array_repr(learning_res[1][:, 0], precision=3, suppress_small=True)))
    logging.info('validate score of root labels is {!r}'.format(
        numpy.array_repr(learning_res[2][:, 0], precision=3, suppress_small=True)))
    logging.info('training score of not-root labels is {!r}'.format(
        numpy.array_repr(learning_res[1][:, 1], precision=3, suppress_small=True)))
    logging.info('validate score of not-root labels is {!r}'.format(
        numpy.array_repr(learning_res[2][:, 1], precision=3, suppress_small=True)))
    return(learning_res)


def inspect_profile(data, proc_func, filename=None):
    if filename is None:
        filename = 'recursive.prof'
    exec_func = partial(proc_func, (data,),
            dict([('random_state', cmd_args.seed),
                  ('learning_rate', cmd_args.lr)]),
            dict([('examples', nrows),
                  ('batch_size', cmd_args.bsize)]))
    cProfile.run('result = exec_func()', filename)
    logging.info("profile %s with #%d-samples " %(proc_func.__name__,
        len(data)))
    prof_stat = pstats.Stats()
    prof_stat.strip_dirs().sort_stats(-1).print_stats()


def proc_train_test_split(y, **kwargs):
    """
    @param cv_class the class used for cross-validation 
    @param is_stratified 

    @return indices of training and test set in tuple and a iterator which will
    yield the indices for each validation set
    """
    presplit = kwargs.pop('predefine', False)
    if presplit:
        predefine = [('n_train', 8544), ('n_dev', 1101), ('n_test', 2210)]
        logging.info("using predefine split #train={n_train} #valid={n_dev} "
                "#test={n_test}".format(**dict(predefine)))
        sizes = numpy.asarray(list(accumulate(map(itemgetter(1), predefine),
            operator.add)))
        train_idx = numpy.arange(sizes[-1])
        return(train_idx[:sizes[0]], train_idx[sizes[0]:sizes[1]], 
                train_idx[sizes[1]:sizes[-1]])
    else:
        indices = None 
        splitter = ShuffleSplit(**kwargs)
        for full_train, test in splitter.split(y, y):
            for train, valid in splitter.split(full_train, full_train):
                indices = (train, valid, test)
        logging.info("using predefine split #train={n_train} #valid={n_dev} "
                "#test={n_test}".format(n_train=len(indices[0]),
                n_dev=len(indices[1]), n_test=len(indices[-1])))
        return(indices)


def get_root_label(tree_strs):
    """
    a quick way to get root label
    """
    return([int(s.strip()[1]) for s in tree_strs])


if __name__ == '__main__':
    from functools import partial
    import argparse
    import cProfile
    import pstats
    import os
    import time

    parser = argparse.ArgumentParser()
    parser.add_argument("subcommand", help='[learnig]')
    parser.add_argument("-f", "--file", dest='file', action="store_const",
            const=os.path.join(os.path.dirname(__file__), '../data/trees.bin'),
            help="supplying cvs file for tree-based phrases")
    parser.add_argument("-n", "--nrows", dest='nrows', action="store",
            default=-1, type=int,
            help="supplying how many examples readin from file")
    parser.add_argument("-b", "--batch_size", dest='bsize', action="store",
            default=1, type=int,
            help="the size of batch for the mini-batch training")
    parser.add_argument("-s", "--seed", dest='seed', action="store",
            default=42, type=int,
            help="the seed used to create random generator")
    parser.add_argument("-e", "--encoder-only", dest='enc', action="store_const",
            const=True, help="only train autoencoder")
    parser.add_argument("-l", "--learning_rate", dest='lr', action="append",
            type=float, required=True,
            help="the learning rate used for gradient descent method")
    parser.add_argument("--predefine", action="store_true", help="using predefined split")
    parser.add_argument("--preproc", dest='preproc', action="store_const",
            const=os.path.join(os.path.dirname(__file__), 
                '../data/treebased_phrases_vocab.model'),
            help="supplying model file for vocabulary mapping to word index")

    cmd_args = parser.parse_args()

    nrows = cmd_args.nrows
    with proc_data(cmd_args.file) as stream:
        tree_strs = stream
        if nrows == -1:
            nrows = len(tree_strs)

        true_root =  get_root_label(tree_strs[:nrows])
        train, valid, test = proc_train_test_split(true_root,
                predefine=cmd_args.predefine)
        raw_data = numpy.asarray(tree_strs[:nrows])

        # preprocess
        preproc = load_preproc(pretrain_loc=cmd_args.preproc)                                             
                                              
        train_features, train_targets = preprocess(preproc, (raw_data[train],),
                sorter__is_order=True)
        valid_features, valid_targets = preprocess(preproc, (raw_data[valid],),
                sorter__is_order=False)
        init_emb = \
            preproc.get_params()['indexer__func'].preprocessor.named_steps[
                    'decomposer'].components_.T
        vocab_size, n_components = init_emb.shape
        assert(cmd_args.lr is not None)
        if not hasattr(cmd_args.lr, '__len__'):
            lr_rates = [cmd_args.lr]
        else:
            lr_rates = cmd_args.lr.copy()
        finetune_rate = lr_rates.pop() 
        pretrain_rate = finetune_rate
        if len(lr_rates) > 0:
            pretrain_rate = lr_rates.pop()
        classifier = construct_model(n_components=n_components,
                vocab_size=vocab_size - 1,  # minus one since blocks will
                                            # automatically add
                random_state=cmd_args.seed,
                learning_rate=finetune_rate)
        classifier.set_params(learning_rate=pretrain_rate, 
                autoencoder__model={'/treeop_wrapper/embed.W':init_emb})
        fit_params = {
                'examples': len(train),
                'batch_size': cmd_args.bsize,
                'log_backend': 'python'
                }
        learning_res = proc_learning(classifier, (train_features,
            train_targets), (valid_features, valid_targets),
            fit_params=fit_params)
        test_features, test_targets = preprocess(preproc, (raw_data[test],),
                sorter__is_order=False)
        acc = classifier.score(test_features, test_targets, 'error_rate', examples=len(test),
                batch_size=1)
        logging.info("%s test with #%d-samples and testing accuracy(root, "
                "not_roots)=(%.3f, %.3f)" %(str(classifier), len(test), *acc))
