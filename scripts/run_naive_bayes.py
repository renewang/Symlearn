from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import (StratifiedKFold, GroupKFold, ShuffleSplit,
        GridSearchCV, learning_curve, ParameterGrid, PredefinedSplit, 
        RandomizedSearchCV)
from sklearn.decomposition import TruncatedSVD
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.utils.validation import check_is_fitted
from sklearn.base import clone 
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.ensemble import ExtraTreesClassifier

from contextlib import contextmanager
from functools import partial
from itertools import accumulate
from collections import OrderedDict

from stanfordSentimentTreebank import create_vocab_variants
from symlearn.utils import (VocabularyDict, count_vectorizer, construct_score, 
    inspect_and_bind)


import h5py
import joblib
import pandas
import numpy
import scipy


import pathlib
import inspect
import logging
import operator
import argparse
import time
import os
import gc


try:
    from _aux import (transform_features, group_fit, labels_to_attributes,
        process_joint_features)
except: 
    logging.info('using pyximport')
    import pyximport
    pyximport.install(inplace=True)
    from _aux import (transform_features, group_fit, labels_to_attributes, 
        process_joint_features)   

FORMAT='%(asctime)s:%(levelname)s:%(threadName)s:%(filename)s:%(lineno)d:%(funcName)s:%(message)s'
logging.captureWarnings(True)
logging.basicConfig(format=FORMAT, datefmt='%m/%d/%Y %I:%M:%S %p',
        level=logging.DEBUG, handlers=[logging.StreamHandler()])

data_dir = os.path.join(os.getenv('DATADIR', default='..'), 'data')

from joblib.numpy_pickle import NumpyUnpickler
class RestrictedUnpickler(NumpyUnpickler):
    """
    overwrite find_class to restrict global modules import
    """

    def find_class(self, module, name):
        import pickle, sys
        # Only allow safe classes from builtins.
        if not (module in globals() or module in sys.modules):
            if name in ['count_vectorizer', 'WordNormalizer', 'VocabularyDict',
            'construct_score', 'inspect_and_bind']:
                logging.warn('skipping importing rquired module %s because not found' % module)
                module = 'symlearn.utils'
            else:
                raise pickle.UnpicklingError("global '%s.%s' is forbidden" %
                    (module, name))
        return(super(RestrictedUnpickler, self).find_class(module, name))
 


def construct_random_ensemble(**kwargs):
    """
    create an bootstrap-based ensemble 
    """
    params = {'n_estimators': 30,         # number of estimators used 
              'criterion'   : 'entropy',
              'bootstrap'   : True,       # indicates if using boostrap sample
              'oob_score'   : True,       # indicates if using out of bag
                                          # sample
                                          # out of bag estimation
              'verbose'     : 10,         # verbosity on training
              'warm_start'  : False,      # indicates if using the previous
                                          # result to for new samples    
              'n_jobs'      : 2           # how many parallel jobs             
            }
    sig = inspect.signature(ExtraTreesClassifier)
    params.update({k: v for k, v in kwargs.items() if k in sig.parameters})
    estimator = ExtraTreesClassifier(**params)
    return(estimator)


def construct_boost_ensemble(**kwargs):
    """
    create an adaboost-based ensemble 
    """
    params = {'loss'         : 'deviance',         # loss function: "deviance" is
                                                   # cross-entropy used in logistic
                                                   # regression and "exponential" is
                                                   # AdaBoost like loss
              'learning_rate': 0.1,                # use to avoid overfitting by
                                                   # trading-off with n_estimators
              'n_estimators' : 10,                 # number of estimators used
              'subsample'    : 1.0,                # sample for gradient if
                                                   # less than 1.0 and SGD and
                                                   # out of bag estimation will
                                                   # be conducted 
              'init'         : None,               # BaseEstimator with fit and predict
                                                   # methods, 
              'verbose'      : 10,                 # verbosity on training
              'warm_start'   : False               # indicates if using the previous
                                                   # result to for new samples              
              }
    sig = inspect.signature(GradientBoostingClassifier)
    params.update({k: v for k, v in kwargs.items() if k in sig.parameters})
    estimator = GradientBoostingClassifier(**params)
    return(estimator)


def train_test_split(n_samples, presplit=False, **kwargs):
    """
    function to make train or test split by random or by the predefine split

    Parameters
    ----------
    @param n_samples: int
        used to specify how many samples are used to create train and test set
        if not all used 
    @param  presplit: boolean
        useed to indicate if predefine split is used; if True, then using the
        default predefine split
    @param kwargs: dict
        available keyword are otherwise, random split will be condcuted
        keywords used in sklearn.cross_validation.ShuffleSplit
    """
    presplit = kwargs.pop('predefine', False)
    predefine = [('n_train', 8544), ('n_dev', 1101), ('n_test', 2210)]
    if presplit:
        logging.info("using predefine split #train={n_train} #valid={n_dev} "
                "#test={n_test}".format(**dict(predefine)))
        sizes = numpy.asarray(list(accumulate(map(operator.itemgetter(1), predefine),
            operator.add)))
        ttl_idx = numpy.arange(sizes[-1])
        return(ttl_idx[:sizes[0]], ttl_idx[sizes[0]:sizes[1]], 
                ttl_idx[sizes[1]:sizes[-1]])
    else:
        if(not "test_size" in kwargs):
            kwargs["test_size"] = 0.15
        if(not "n_splits" in kwargs):
            kwargs["n_splits"] = 1
        ttl_idx = numpy.arange(n_samples)
        # keeping same test
        test = ttl_idx[-1*predefine[-1][-1]:] 
        indices = None 
        splitter = ShuffleSplit(**kwargs)
        for train, valid in splitter.split(ttl_idx[:-1*predefine[-1][-1]]):
            indices = (train, valid, test)
        logging.info("using random split #train={n_train} #valid={n_dev} "
                "#test={n_test}".format(n_train=len(indices[0]),
                n_dev=len(indices[1]), n_test=len(indices[-1])))
        return(indices)


@contextmanager
def gc_context():
    """
    experimental code to try context for gc debugging  invocation
    """
    if gc.isenabled():
        gc.disable()
    yield
    if not gc.isenabled():
        gc.enable()
        

def construct_decision_tree(**kwargs):
    """
    providing default parameters for construct a scikit-learn decision tree
    """
    tree_kws = {'criterion': 'entropy',           # measure split. options: "gini" impurity and cross
                                                  # "entropy"
                'splitter': 'best',               # algorithm for split a node. options: best split and random split
                'max_features': 'auto',           # criterion for split features. 
                                                  # options: None for no split using n_features
                                                  # auto / sqrt: taking sqrt
                                                  # log2: taking log2 
                'max_depth': None,                # the max depth for tree: None for splitting till all leaves are pure
                                                  # this can work with the min_sample_split and turn off 
                                                  # if max_leaf_node is not None
                'min_samples_split': 100,         # the minimal number of samples to split a node
                'min_samples_leaf': 100,          # the minimal sample required to be in leaf node
                'min_weight_fraction_leaf': 0.05, # the fraction of samples required to be in the leaf
                'max_leaf_nodes': None,           # the maximal number of leaf nodes to be grown
                'class_weight': 'balanced',       # the weight to balance target classes
                'random_state': None,             # random seed
                'presort'     : 'auto'            # indicates if using sorting
                                                  # to speed up
               }
    tree_kws.update(kwargs)
    tree_ba = inspect_and_bind(DecisionTreeClassifier, **tree_kws)
    estimator = DecisionTreeClassifier(*tree_ba.args, **tree_ba.kwargs)
    return(estimator)


def construct_preprocessor(**kwargs):
    """
    providing default construction of preprocessor
    """
    new_kwargs = {
        'strip_accents': 'unicode',
        'tokenizer': simple_split,
        'analyzer': 'word',
        # work-around for segfault,
        # https://github.com/scikit-learn/scikit-learn/issues/7626
        'algorithm': 'arpack',
        'sublinear_tf': True,
        'use_idf': True}
    new_kwargs.update(kwargs)

    vocab = kwargs.get('vocabulary', None)
    if not vocab: # compute from training set
        vectorizer_ba = inspect_and_bind(CountVectorizer, **new_kwargs)
        extractor = CountVectorizer(*vectorizer_ba.args, **vectorizer_ba.kwargs)
    else:
        new_kwargs.update({'vocab_file': vocab})
        extractor_ba = inspect_and_bind(count_vectorizer, **new_kwargs)
        extractor = FunctionTransformer(count_vectorizer(*extractor_ba.args,
            **extractor_ba.kwargs), validate=False)
        
    transformer_ba = inspect_and_bind(TfidfTransformer, **new_kwargs)
    decomposer_ba = inspect_and_bind(TruncatedSVD, **new_kwargs)

    preprocessor = Pipeline([
    ('vectorizer', extractor),
    ('transformer', TfidfTransformer(*transformer_ba.args,
        **transformer_ba.kwargs)),
    ('decomposer', TruncatedSVD(*decomposer_ba.args, **decomposer_ba.kwargs))])
    return(preprocessor)
 

def preload_helper(csv_file, vocab=None, pretrain_loc=None, n_rows=-1,
    pre_split=False):
    """
    load pretrain features and construct models
    """
    if pretrain_loc and os.path.exists(pretrain_loc):
        from unittest.mock import patch
        with patch.object(joblib.numpy_pickle, 'NumpyUnpickler', 
            new=RestrictedUnpickler, spec_set=True):
            preproc = joblib.load(pretrain_loc)
        vocab = preproc.named_steps['vectorizer'].func.vocab
    elif vocab:
        vocab = VocabularyDict(vocab)
        preproc = construct_preprocessor(random_state=seed, vocabulary=vocab)
    else:
        vocab, preproc = None, None

    features = transform_features(csv_file, n_rows=n_rows, preproc=preproc,
                vocab=vocab)

    n_samples = len(features['wordtokens'])
    targets = numpy.hstack(list(map(lambda x: x.data[x.mask], 
        features['sentiments']))) # taking out root label

    # construct train / test split
    train, valid, test = train_test_split(n_samples, predefine=pre_split)

    train_features = pandas.DataFrame({
        'ids': features['ids'][train], 
        'levels': features['levels'][train], 
        'phrases': features['phrases'][train], 
        'sentiments': features['sentiments'][train]})
    valid_features = pandas.DataFrame({
        'ids': features['ids'][valid], 
        'levels': features['levels'][valid], 
        'phrases': features['phrases'][valid], 
        'sentiments': features['sentiments'][valid]})
    test_features = pandas.DataFrame({
        'ids': features['ids'][test], 
        'levels': features['levels'][test], 
        'phrases': features['phrases'][test], 
        'sentiments': features['sentiments'][test]})

    if preproc:
        logging.info("preprocessor is loaded from %s using n_components=%d n_words=%d" %
            (pretrain_loc, preproc.named_steps['decomposer'].n_components, 
                len(preproc.named_steps['vectorizer'].func.vocab)))

    return preproc, (train_features, targets[train]), \
        (valid_features, targets[valid]), (test_features, targets[test])


def construct_naive_bayes(preprocessor=None, **kwargs):
    """
    providing default construction for naive bayes
    """
    if preprocessor is None:
        preprocessor = construct_preprocessor(**kwargs)
        estimator = Pipeline([
            ('preprocessor', preprocessor),
            ('normalizer', kwargs.get('normalizer', None)),
            ('classifier', kwargs.get('classifier', None))])
    else:
        estimator = Pipeline([
            ('normalizer', kwargs.get('normalizer', None)),
            ('classifier', kwargs.get('classifier', None))])
    return(estimator)


def simple_split(x):
    return(x.split());


def construct_model(weights, cvkwargs, gridkwargs, preproc=None):
    """
    providing default construction for the classifier models mainly for weights
    selection in cross-validation settings

    Parameters
    ----------
    @param weights: dict
        a dict storing differen sample weighting schemes each entry providing
        the name of weighting scheme and numpy.ndarray which contain weights
        for each training samples
    @param cvkwargs: dict
        additional keywords will be passed into scikit-learn cross-validaiton
        instance
    @param gridkwargs: dict
        additional keywords will be passed into scikit-learn GridSearchCV
        instance 
    @param preproc: scikit-learn BaseEstimator or TransformMixin instance
        instance used to transform text features into numerical ones
    """
    base_dict = {}
    if not preproc:
        base_dict = {'preprocessor__vectorizer__ngram_range': [(1, 1)], 
                'preprocessor__decomposer__n_components':
                numpy.linspace(50, 100, 1, dtype=numpy.int)}
    params = [
            {'normalizer': [StandardScaler()], 'classifier': [GaussianNB()]},
            # comment out for now. Code here is just for comparison and might
            # not be really useful 
            #{'normalizer': [SoftMaxScaler(scale_func='sigmoid')], 
            #    'classifier': [MultinomialNB()],
            #    'classifier__alpha': numpy.logspace(-2, 0, 1)}
            ]
    for param in params:
        param.update(base_dict)

    estimator = construct_naive_bayes(preproc)
    searchers = OrderedDict()
    # sentence only
    no_sentid_cv = StratifiedKFold(**cvkwargs)
    searchers['sentence_only'] = GridSearchCV(estimator=clone(estimator),
            param_grid=params, scoring=make_scorer(),
            cv=no_sentid_cv, **gridkwargs)

    # split based on the y-labels without considering sentence ids
    searchers['no_sentid'] = GridSearchCV(estimator=clone(estimator),
            param_grid=params, scoring=make_scorer(construct_score),
            cv=no_sentid_cv, **gridkwargs)

    sentid_cv = GroupKFold(n_splits=cvkwargs.get('n_splits'))
    # cv based on setence_id without weighting
    searchers['no_weight_sentid'] = GridSearchCV(estimator=clone(estimator),
            param_grid=params, scoring=make_scorer(construct_score),
            cv=sentid_cv, **gridkwargs)

    # cv based on setence_id with equal weighting
    searchers['weight_by_size'] = GridSearchCV(estimator=clone(estimator),
            param_grid=params, scoring=make_scorer(construct_score),
            cv=sentid_cv, fit_params={
                'classifier__sample_weight': weights['weight_by_size']},
            **gridkwargs)

    # cv based on setence_id with tree_size weighting
    searchers['weight_by_node'] = GridSearchCV(estimator=clone(estimator),
            param_grid=params, scoring=make_scorer(construct_score),
            cv=sentid_cv, fit_params={
                'classifier__sample_weight': weights['weight_by_node']},
            **gridkwargs)
    return(searchers)


class _fit_and_score(object):

    def __call__(self, estimator, X, y, scorer, train, test, verbose,
                   parameters, fit_params, return_train_score=False,
                   return_parameters=False, return_n_test_samples=False,
                   return_times=False, error_score='raise'):
        """
        borrow from sklearn.model_selection._fit_and_score in order to 
        return oob_score for ensemble classifier

        """
        from sklearn.utils.metaestimators import _safe_split
        from sklearn.model_selection._validation import _score
        if verbose > 1:
            if parameters is None:
                msg = ''
            else:
                msg = '%s' % (', '.join('%s=%s' % (k, v)
                              for k, v in parameters.items()))
            print("[CV] %s %s" % (msg, (64 - len(msg)) * '.'))

        # Adjust length of sample weights
        fit_params = fit_params if fit_params is not None else {}
        fit_params = dict([(k, _index_param_value(X, v, train))
                          for k, v in fit_params.items()])

        if parameters is not None:
            estimator.set_params(**parameters)

        start_time = time.time()

        X_train, y_train = _safe_split(estimator, X, y, train)
        X_test, y_test = _safe_split(estimator, X, y, test, train)

        try:
            if y_train is None:
                estimator.fit(X_train, **fit_params)
            else:
                estimator.fit(X_train, y_train, **fit_params)

        except Exception as e:
            # Note fit time as time until error
            fit_time = time.time() - start_time
            score_time = 0.0
            if error_score == 'raise':
                raise
            elif isinstance(error_score, numbers.Number):
                test_score = error_score
                if return_train_score:
                    train_score = error_score
                warnings.warn("Classifier fit failed. The score on this train-test"
                              " partition for these parameters will be set to %f. "
                              "Details: \n%r" % (error_score, e), FitFailedWarning)
            else:
                raise ValueError("error_score must be the string 'raise' or a"
                                 " numeric value. (Hint: if using 'raise', please"
                                 " make sure that it has been spelled correctly.)")

        else:
            fit_time = time.time() - start_time
            test_score = _score(estimator, X_test, y_test, scorer)
            score_time = time.time() - start_time - fit_time
            if return_train_score:
                if hasattr(estimator, 'oob_score_'):
                    logging.debug('out of bag score is used to compute train_score '
                        'instead of {} providing'.format(str(scorer)))
                    train_score = estimator.oob_score_
                else:
                    train_score = _score(estimator, X_train, y_train, scorer)

        if verbose > 2:
            msg += ", score=%f" % test_score
        if verbose > 1:
            total_time = score_time + fit_time
            end_msg = "%s, total=%s" % (msg, joblib.logger.short_format_time(total_time))
            print("[CV] %s %s" % ((64 - len(end_msg)) * '.', end_msg))

        ret = [train_score, test_score] if return_train_score else [test_score]

        if return_n_test_samples:
            ret.append(_num_samples(X_test))
        if return_times:
            ret.extend([fit_time, score_time])
        if return_parameters:
            ret.append(parameters)
        return ret


def parallel_fit(estimator, X, y=None, prealloc=None,  parameters={},
        fit_params={}):
    """
    Parameters
    ----------
    @param estimator: sickit-learn base estimator or its subclass instance
       the estimator has "fit" method for training 
    @param X: numpy array
       training features will pass into estimator's fit method
    @param y: numpy array
       training targets will pass into estimator's fit method
    @param prealloc: numpy memmap file object
       used to store the fitting result 
    @param parameters: dict
       used to set estimator parameters
    @param fit_params: dict
       used to pass additional parameters into estimator's fit method
    """
    assert(parameters)
    estimator.set_params(**parameters)
    estimator.fit(X, **fit_params)
    ret = (estimator.explained_variance_, estimator.explained_variance_ratio_)
    logging.debug("complete traning for {n_components}".format(**parameters))
    if prealloc is not None:
        if len(ret[0]) == prealloc.shape[1] or numpy.any(
                prealloc[0, :len(ret[0])] != ret[0]):
            prealloc[0, :len(ret[0])] = ret[0] # variation 
            prealloc[1, :len(ret[1])] = ret[1] # variation ratio
            prealloc.flush()
        else:
            logger.warn("overwritten prealloc {}".format(parameters))
        return
    else:
        return ret


def cal_learning_curve(clf, train_features, train_targets, **kwargs):
    """
    a simple wrapper function to supplement to the scikit-learn learning_curve
    module in order to customize inputs
    Parameters
    ----------
    @param clf: scikit-learn BaseEstimator instance
        needs to have predict or scores methods for performance evaluation
    @param train_features: numpy.ndarray, scipy.sparse.spmatrix or list
        providing the features for training
    @param train_targets: numpy.ndarray or list
        providing the true labels for performance evaluation
    @param kwargs: dict
        additional keywords passed into scikit-learn learning_curve function
    """

    from unittest.mock import patch 
    from sklearn.model_selection import _validation

    samplesizes = kwargs.get("train_sizes") 
    learning_func = kwargs.pop("learn_func", learning_curve)
    if isinstance(train_features, (numpy.ndarray, scipy.sparse.spmatrix)):
        data_ = train_features
    elif isinstance(train_features, (pandas.DataFrame, pandas.Series)):
        data_ = train_features.values 
    elif isinstance(train_features, (dict)): 
        data_ = pandas.DataFrame(train_features).values
    ttl_size = (data_.size * data_.dtype.itemsize * numpy.max(samplesizes)) / (1024**2)
    logging.info('start training with training size %d (%.2f MB in memory)' % (
            data_.shape[0] * numpy.max(samplesizes), ttl_size))
    with gc_context() as gc_siwtch:
        start_time = time.time()
        try:
            with patch.object(_validation, '_fit_and_score',  new_callable=_fit_and_score):
                learning_res = learning_curve(clf, train_features, train_targets, **kwargs)
        except TypeError as e:
            if isinstance(train_features, scipy.sparse.spmatrix):
                with patch.object(_validation, '_fit_and_score',  new_callable=_fit_and_score):
                    learning_res = learning_curve(clf, train_features.toarray(),
                        train_targets, **kwargs)
            else:
                raise e
        end_time = time.time()
    logging.info('complete training in total time %.2f seconds' % (end_time - start_time))
    logging.info('training score is {!r}'.format(
            numpy.array_repr(learning_res[1].mean(axis=1), precision=3, suppress_small=True)))
    logging.info('validate score is {!r}'.format(
            numpy.array_repr(learning_res[2].mean(axis=1), precision=3, suppress_small=True)))
    return(learning_res)


def construct_multiple_classifiers(preproc, train_features, train_targets,
                                   cvkwargs, gridkwargs, 
                                   classifier_maker=construct_naive_bayes,
                                    **kwargs):
  """
  group phrases into levels and test on their accuracy with sentence only classifier to see 
    1. if there’s need to re-construct vocabulary set and truncated SVD 
    2. if there’s enough training examples for each level
  """
  max_level = kwargs.pop('max_level')
  init_est = kwargs.pop('init_est', StandardScaler())
  init_clf = kwargs.pop('init_clf', GaussianNB())
  n_classes = kwargs.pop('n_classes', 5)
  
  search_name = kwargs.pop('searcher_name', 'GridSearchCV')

  if search_name == 'GridSearchCV':
    search_maker = partial(GridSearchCV, param_grid={'classifier__priors': [
            (1/n_classes) * numpy.ones(n_classes),  # uniform
            *(numpy.asarray(arr)/numpy.sum(arr) for arr in [
            [0.8, 0.05, 0.05, 0.05, 0.05], # class 0 dominating
            [0.05, 0.8, 0.05, 0.05, 0.05], # class 1 dominating
            [0.05, 0.05, 0.8, 0.05, 0.05], # class 2 dominating
            [0.05, 0.05, 0.05, 0.8, 0.05], # class 3 dominating
            [0.05, 0.05, 0.05, 0.05, 0.8], # class 4 dominating
            ])
        ]})
  else:
    search_maker = partial(RandomizedSearchCV, param_distributions={
        'classifier__priors': [scipy.stats.dirichlet(alpha=numpy.ones(n_classes))]
        })

  estimators = [search_maker(
    estimator=classifier_maker(preprocessor=preproc, normalizer=clone(init_est), 
        classifier=clone(init_clf)), **gridkwargs, cv=StratifiedKFold(**cvkwargs))
        for _ in range(max_level)]
  logging.info("%d label estimators are constrcuted and start training ..." % max_level)
  estimators = group_fit(train_features, preproc, estimators, max_level) 
  logging.info("complete %d label estimators ..." % max_level)
  return estimators


def process_word_features(csv_file, **kwargs):
    """
    process word features and smooth out the vocabulary set by collecting rare
    words

    Parameters
    ----------
    @param csv_file: str
        file to store parsing tree information including tree_id, node_id,
        phrases, sentiments, start_pos and end_pos
    @param kwargs: dict
        additional parameters used to pass into other functions called
    """
    seed = kwargs.pop('seed', 42)
    pre_split = kwargs.pop('predefine', None)
    n_rows = kwargs.pop('n_rows', None)
    n_words = kwargs.pop('n_words', 4500)
    n_features = kwargs.pop('n_features', 300)
    store_input = kwargs.pop('store_input', False)
    cutoff_ = 0.9 # keep 90% variance explained
    
    # handle data
    ids, sentences, sentiments, levels, weights, _ = \
            preprocess_data(csv_file, n_rows=n_rows)
    train, valid, test = train_test_split(numpy.max(ids) + 1,
        predefine=pre_split)

    # handle vocabulary
    files = [fn for fn in pathlib.Path(data_dir).glob('*_vocab*') 
        if fn.suffix not in ['.dat', '.hdf5', '.model']]
    if not files:
        gen = create_vocab_variants(csv_file, n_rows=n_rows)
    else:
        gen = iter(map(lambda x: x.as_posix(), files))

    while True:
        try:
            vocab = next(gen)
        except StopIteration as e:
            break
        
        # construct model
        preprocessor = construct_preprocessor(random_state=seed,
                vocabulary=vocab, max_features=n_words, 
                algorithm='randomized', n_components=n_features)
        X = sentences[train].tolist()
        for name, trans in preprocessor.steps[:-1]:
          X = trans.fit_transform(X)
                
        # compute the total size needs to be allocated for the return values which
        # is the sum of n_components used for all params
        decomposer_ = clone(preprocessor.steps[-1][-1])
        mmap_fn = os.path.join(data_dir, "%s.dat"%(vocab))
        if os.path.exists(mmap_fn):
            logging.info('%s already exists! Skip to avoid being overwritten' %
                mmap_fn)
            decomposer_.fit(X)
            acc_explained_ratio_ = numpy.add.accumulate(decomposer_.explained_variance_ratio_)
            if acc_explained_ratio_[-1] <= cutoff_:
                cutoff_ = acc_explained_ratio_[-1] - 0.1
            # the best_n_components whose accumulative explained_ratio should be greater than 
            # cutoff_ and the slope of accumulative explained_ration should be close to zero
            sel_mask = numpy.logical_and(acc_explained_ratio_ > cutoff_, 
                numpy.isclose(decomposer_.explained_variance_ratio_, 0., atol=1e-4))
            if numpy.all(sel_mask) == False:
                preprocessor.set_params(decomposer = decomposer_)
            else:
                cur_best_n = numpy.min(numpy.arange(decomposer_.n_components)[sel_mask])
                preprocessor.steps[-1][-1].set_params(n_components=cur_best_n)
                preprocessor.steps[-1][-1].fit(X)
            joblib.dump(preprocessor, '%s.model' %(vocab))
            continue
        
        # construct a base estimator
        max_n_components = numpy.min(X.shape)
        params = {'n_components': [int(n) for n in [max_n_components]],
                  'algorithm': ['randomized', 'arpack']}
        logging.info('processing %s ...'%(vocab))
        
        if store_input:
            logging.info('store input in %s.hdf5'%(os.path.splitext(mmap_fn)[0]))
            with h5py.File("%s.hdf5" % (os.path.splitext(mmap_fn)[0]), mode='w') as root:
                # store parameters
                root.create_dataset("n_components",
                        data=numpy.asarray(params['n_components']))
                # store inputs
                input_ = root.create_group("input")
                ds = input_.create_dataset("csr_attrs", shape=(2,),
                        dtype=h5py.special_dtype(vlen=X.indices.dtype))
                ds.attrs['shape'] = numpy.asarray(X.shape)
                ds[0] = X.indices
                ds[1] = X.indptr
                input_.create_dataset("csr_data", data=X.data)


        with open(mmap_fn, "w+b") as mmap_fp:
            # preallocate the shared memmap files to store the result
            Xtvars = numpy.memmap(mmap_fp, dtype=numpy.float, mode="w+",
                    shape=(2, max_n_components))
            # ensure the local modifications are all written to disk
            Xtvars.flush()
            # don't use n_jobs=-1 on OSX; otherwise process freezes
            # disable joblib automatically dump memmap file by assinging
            # max_nbytes=None 
            parallel = joblib.Parallel(n_jobs=2, pre_dispatch='2 * n_jobs', verbose=100,
                    max_nbytes=None)
            logging.info('start joblib parallel job {}'.format(parallel))
            joblib.parallel(joblib.delayed(parallel_fit)(clone(decomposer_), Xt, None,
                Xtvars, param) for param in ParameterGrid(params))
            logging.info('complete parallel fitting and store in %s' % mmap_fn)


def run_single_classifier(csv_file, cvkwargs, gridkwargs, **kwargs):
    batch_mode = kwargs.pop('batch_mode', False)
    n_rows = kwargs.pop('n_rows', -1)
    use_ensemble = kwargs.pop('use_ensemble', False)
    pre_split = kwargs.pop('predefine', False)
    model_name = 'decision_tree_single.model'
    if use_ensemble:
        classifier_maker = construct_random_ensemble
        model_name = 'ensemble_%s'%(model_name)
    else:
        classifier_maker = construct_decision_tree

    preproc, (train_features, train_targets), (valid_features, valid_targets), \
        (test_features, test_targets) = preload_helper(csv_file, n_rows=n_rows,
            pre_split=pre_split)

    # construct models
    vectorizer = DictVectorizer(dtype=numpy.float32, sparse=True)
    tree_clf = classifier_maker(**kwargs) 

    # vectorize features
    train_mats = process_joint_features((train_features['sentiments'].tolist(), train_features), 
        vectorizer=vectorizer)
    valid_mats = process_joint_features((valid_features['sentiments'].tolist(), valid_features),
        vectorizer=vectorizer)
    train_mats = scipy.sparse.vstack([train_mats, valid_mats])
    train_targets = numpy.hstack([train_targets, valid_targets])

    if pre_split:
        train = len(train_targets) - len(valid_targets)
        test_fold = -1 * numpy.ones(len(train_targets), dtype=numpy.int8)
        test_fold[train:] = 0
        # sorting the features and targets with their length
        unsort_train_mats = train_mats.tolil(copy=True)
        train_mats = scipy.sparse.lil_matrix(train_mats.shape, dtype=train_mats.dtype)
        sort_by_size = numpy.argsort(unsort_train_mats[:train].sum(axis=1).A1)
        train_mats[:train] = unsort_train_mats[:train][sort_by_size]
        train_mats[train:] = unsort_train_mats[train:]
        train_mats = train_mats.tocsr()
        del unsort_train_mats
        assert(numpy.all(numpy.diff(train_mats[:train].sum(axis=1).A1) >= 0))
        unsort_targets = train_targets.copy()
        train_targets[:train] = unsort_targets[:train][sort_by_size]
        train_targets[train:] = unsort_targets[train:]
        del unsort_targets
        gridkwargs.update({'cv': PredefinedSplit(test_fold)})
    else:
        gridkwargs.update({'cv': StratifiedKFold(**cvkwargs)})

    learning_res = cal_learning_curve(tree_clf, train_mats, train_targets,
            train_sizes=numpy.linspace(0.1, 1.0, 50),
            **gridkwargs)

    # refit with all training data and test
    test_mats = process_joint_features((test_features['sentiments'].tolist(), test_features), 
        vectorizer=vectorizer)
    assert(train_mats.shape[-1] == test_mats.shape[-1])
    tree_clf.fit(train_mats, train_targets)
    try:
        predictions = tree_clf.predict(test_mats)
    except TypeError as e:
        if isinstance(test_mats, scipy.sparse.spmatrix):
            predictions = tree_clf.predict(test_mats.toarray())
        else:
            raise e
    logging.info('completing fitting and test on test set with {:.4f}'
            ' accuracy'.format(
                numpy.mean(predictions == test_targets)))
    if(hasattr(tree_clf, 'steps')):
        logging.info('tree classifier has nodes {}'.format(
            tree_clf.steps[-1][-1].tree_.node_count))

    if batch_mode:
        return (learning_res, numpy.mean(predictions == test_targets))

    joblib.dump((tree_clf, vectorizer, preproc, (train_features, train_targets), 
            (valid_features, valid_targets), (test_features, test_targets)), 
            os.path.join(data_dir, model_name))
    logging.info('dumping {}'.format(model_name))


def run_multi_classifiers(csv_file, cvkwargs, gridkwargs, **kwargs):
    """
    two-tiered classifiers
    """
    max_level = kwargs.pop('max_level')
    pre_split = kwargs.pop('predefine', False)
    seed = kwargs.pop('random_state', None)
    n_rows=kwargs.pop('n_rows', -1)
    n_classes = kwargs.pop('n_classes', 5)
    use_ensemble = kwargs.pop('use_ensemble', False)
    pretrain_loc = kwargs.pop('pretrain_model',
            os.path.join(data_dir, 'treebased_phrases_vocab.model'))
    vocab = kwargs.pop('vocab', os.path.join(data_dir, 'treebased_phrases_vocab'))
    model_name = 'decision_tree_multi.model'
    if use_ensemble:
        model_name = 'ensemble_%s'%(model_name)
        classifier_maker = construct_random_ensemble
    else:
        classifier_maker = construct_decision_tree

    preproc, (train_features, train_targets), (valid_features, valid_targets), \
        (test_features, test_targets) = preload_helper(
            csv_file, vocab=vocab, pretrain_loc=pretrain_loc, pre_split=pre_split)
    
    # calling function which will train n-independent label predictors
    label_searchers = construct_multiple_classifiers(preproc, train_features, train_targets,
        cvkwargs, gridkwargs, max_level=max_level)
    
    # construct the final model
    vectorizer = DictVectorizer(sparse=True)
    classifier = Pipeline(
        [('transformer', None),
         ('classifier', classifier_maker(**kwargs))])

    # iterate different label_predictors (pre-trained with different parameters)
    transformers = []
    for i, param in enumerate(label_searchers[0].cv_results_['params']):
        scores = numpy.asarray([est.cv_results_['mean_test_score'][i]
            for est in label_searchers])
        logging.info('label_predictors fit with priors=%s accuracy=%s' %(
            numpy.array2string(param['classifier__priors'], suppress_small=True, precision=3),
            numpy.array2string(scores, suppress_small=True, precision=3)))
        label_predictors = [clone(est.estimator).set_params(**param) 
                            for est in label_searchers]
        transformers.append(FunctionTransformer(
            labels_to_attributes(preproc, vectorizer), 
            validate=False, kw_args={'label_predictors': label_predictors}))

    if pre_split:
        test_fold = -1 * numpy.ones(len(train_targets) + len(valid_targets), dtype=numpy.int8)
        test_fold[len(train_targets):] = 0
        gridkwargs.update({'cv': PredefinedSplit(test_fold)})
    else:
        gridkwargs.update({'cv': StratifiedKFold(**cvkwargs)})        
    
    # tuning the ensemble of classifiers and then refit with all training data
    searcher = GridSearchCV(classifier, {'transformer': transformers}, **gridkwargs)
    searcher.fit(pandas.concat([train_features, valid_features]), 
        numpy.hstack([train_targets, valid_targets]))
    logging.info('completing fitting and test on test set with {} accuracy'.format(
                searcher.score(test_features, test_targets)))
    joblib.dump((searcher, vectorizer, preproc, label_searchers, \
        (train_features, train_targets), (valid_features, valid_targets), \
        (test_features, test_targets)), os.path.join(data_dir, model_name))
    logging.info('completing fitting and dumping {}.model'.format(model_name))


def run_cross_validation(csv_file, cvkwargs, gridkwargs, **kwargs):
    """
    employ four naive bayes with different sample weights
    """
    classifiers = {'no_sentid_nmb': None, 'no_weight_sentid_nmb': None,
                   'weight_by_size_nmb': None, 'weight_by_node_nmb': None,
                   'sentence_only_nmb': None}
    readables = {
        'no_sentid' : 'no weighting (random cv)',
        'no_weight_sentid' : 'no weighting (sentence based cv)',
        'weight_by_size' : 'uniform weight by phrase size (sentence based cv)',
        'weight_by_node' : 'unequal weight by tree node (sentence based cv)',
        'sentence_only': 'not including partial phrases'}

    pre_split = kwargs.get('predefine', False)
    seed = kwargs.get('random_state', 42)
    n_rows=kwargs.get('n_rows', -1)
    n_classes = kwargs.get('n_classes', 5)
    pretrain_loc = kwargs.get('pretrain_model')
    vocab = kwargs.get('vocab', os.path.join(data_dir,
        'treebased_phrases_vocab'))

    preproc, (train_features, train_targets), (valid_features, valid_targets), \
        (test_features, test_targets) = preload_helper(csv_file, n_rows=n_rows,
            pre_split=pre_split)

    # construct train / test split
    try:
        check_is_fitted(preproc.steps[-1][-1], 'components_')
    except AttributeError:
        preproc.fit(numpy.asarray(wordtokens)[train])
        joblib.dump(preproc, os.path.join(data_dir, 'TruncatedSVD.model'))

    train_mats = preproc.transform(train_features['phrases'])
    train_weights = train_features['weights']
    train_sentiments = train_features['sentiments']
    
    test_mats = preproc.transform(test_features['phrases'])
    test_weights = test_features['weights']
    test_sentiments = test_features['sentiments']

    logging.info('complete transforming input to %d-components word vectors' %
            features.shape[1])
    searchers = construct_model(train_weights, cvkwargs, gridkwargs, preproc)

    for name, searcher in searchers.items():
        logging.info('start GridSearchCV fitting with %s and '
                     'weighting scheme is %s' % (name, readables[name]))
        start_time = time.time()
        if name == 'sentence_only':
            searcher.fit(train_mats[train_features['level']==0], 
                train_sentiments[train_features['level']==0])
        else:
            searcher.fit(train_mats, train_sentiments,
                    groups=train_features['ids'])
        logging.info('complete GridSearchCV fitting with {} (#{} '
                'samples) in {:.2f} seconds'.format(name, len(phrases),
                    time.time() - start_time))
        searchers[name].score(test_mats, test_sentiments)
        classifiers['%s_nmb' % name] = searcher

    # dump classifiers
    model_files = [] 
    for model_name, cls in classifiers.items():
        fn = pathlib.Path(data_dir).joinpath("%s.model" % (model_name))
        joblib.dump(cls, fn)
        model_files.append(fn)
        logging.info('store {}'.format(str(fn)))
    return(model_files)


def main(*args):
    parser = argparse.ArgumentParser()
    parser.add_argument("subcommand",
            help="[weight_training|single_classifier|multi_classifier|preprocess_features]")
    parser.add_argument("-f", "--file", default='treebased_phrases.csv', help="csv file for data")
    parser.add_argument("-n", "--nrows", type=int, default=-1, help="number of rows to read in")
    parser.add_argument("-w", "--nwords", type=int, default=4500, help="size of vocabulary")
    parser.add_argument("-c", "--ncomponents", type=int, default=300, help="maximal number of components used")
    parser.add_argument("-t", "--max_levels", type=int, default=16, help="threadshold for maximal level used")
    parser.add_argument("--predefine", action="store_true", help="using predefined split")
    parser.add_argument("--seed", type=int, default=None, 
        help="specify the seed used for data splitting and modeling")
    parser.add_argument("--ensemble", action="store_true",
            help="using ensemble to construct tree classifier")
    args = parser.parse_args()
    if args.subcommand.startswith('w'):
        logging.info("start weighted cross-validation")
        cvkws = {'n_splits': 10, 'shuffle': True, 'random_state': args.seed}
        gridkws = {'verbose': 10, 'refit': True} 
        run_cross_validation(os.path.join(data_dir, args.file),
                cvkws, gridkws, predefine=args.predefine, n_rows=args.nrows)
    elif args.subcommand.startswith('s'):
        logging.info("start using decision tree to predict sentiment"
                " file: %s", args.file)
        cvkws = {'n_splits': 10, 'random_state':args.seed} # used to create the same partition
        gridkws = {'scoring': 'accuracy', 'verbose': 0}
        run_single_classifier(os.path.join(data_dir, args.file),
                cvkws, gridkws, predefine=args.predefine, n_rows=args.nrows,
                use_ensemble=args.ensemble, random_state=None, # for creating randomness in model
                max_level=args.max_levels)
    elif args.subcommand.startswith('m'):
        logging.info("start using naive bayes + decision tree to predict"
                " sentiment file: %s", args.file)
        cvkws = {'n_splits': 3, 'random_state': args.seed} # used to create the same partition
        gridkws = {'scoring': 'accuracy', 'verbose': 10, 
                   'n_jobs': 2, 'refit': True}
        run_multi_classifiers(os.path.join(data_dir, args.file),
                cvkws, gridkws, predefine=args.predefine, n_rows=args.nrows,
                use_ensemble=args.ensemble, random_state=None,  # for creating randomness in model
                max_level=args.max_levels)
    elif args.subcommand.startswith('p'):
        logging.info("preprocessing and transform words into embedding"
                ", sentiment file: %s", args.file)
        process_word_features(os.path.join(data_dir, args.file),
                n_rows=args.nrows, predefine=args.predefine, 
                n_words=args.nwords, n_features=args.ncomponents)
        logging.info('complete preprocessing word features')


if __name__ == '__main__':
    import sys
    main(*sys.argv[1:])
