from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import (StratifiedKFold, GroupKFold, ShuffleSplit,
        GridSearchCV, learning_curve, ParameterGrid, PredefinedSplit, StratifiedShuffleSplit)
from sklearn.decomposition import TruncatedSVD
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer, brier_score_loss
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.utils.validation import check_is_fitted, _num_samples
from sklearn.base import clone 
from sklearn.preprocessing import FunctionTransformer, StandardScaler, label_binarize
from sklearn.ensemble import (ExtraTreesClassifier, GradientBoostingClassifier, 
    AdaBoostClassifier)
from sklearn.calibration import _SigmoidCalibration as SoftMaxScaler
from sklearn.dummy import DummyClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.multiclass import OneVsRestClassifier
from sklearn.base import BaseEstimator
from contextlib import contextmanager
from functools import partial
from collections import OrderedDict, deque
from stanfordSentimentTreebank import create_vocab_variants
from symlearn.utils import (VocabularyDict, count_vectorizer, construct_score, 
    inspect_and_bind)
from gensim.models.keyedvectors import KeyedVectors
from calibration import CalibratedClassifierCV, CalibratedClassifier
from exec_helper import (patch_pickled_preproc, data_dir, load_from, 
                         LanguageModelPickler)

import h5py
import joblib
import pandas
import numpy
import scipy
import cython

import json
import pathlib
import inspect
import logging
import operator
import argparse
import time
import os
import gc


if cython.compiled:
    from _aux import (transform_features, group_fit, labels_to_attributes, 
        process_joint_features)       
else:
    from aux import (transform_features, group_fit, labels_to_attributes,
        process_joint_features)


FORMAT='%(asctime)s:%(levelname)s:%(threadName)s:%(filename)s:%(lineno)d:%(funcName)s:%(message)s'
logging.captureWarnings(True)
logging.basicConfig(format=FORMAT, datefmt='%m/%d/%Y %I:%M:%S %p',
        level=logging.INFO, handlers=[logging.StreamHandler()])


class DummyProxy(DummyClassifier):
    """
    some customize DummyClassifier behaviors
    """
    def __init__(self, priors=None):
        super(DummyProxy, self).__init__(strategy='stratified')
        self.priors = priors

    def fit(self, X, y, sample_weight=None):
        """
        don't fit data
        """
        super(DummyProxy, self).fit(X, y, sample_weight=sample_weight)
        if self.priors is not None:
            # overwrite
            self.classes_ = numpy.arange(len(self.priors))
            self.n_classes_ = len(self.priors)
            self.class_prior_ = self.priors
        return self


def construct_random_ensemble(**kwargs):
    """
    create an bootstrap-based ensemble 
    """
    params = {'n_estimators': 50,               # number of estimators used 
              'criterion'   : 'entropy',
              'max_features' : 'auto',          # the maximal features for a best split
              'max_depth': None,                # the max depth for tree: None for splitting till all leaves are pure
                                                # this can work with the min_sample_split and turn off 
                                                # if max_leaf_node is not None
              'min_samples_split': 2,           # the minimal number of samples to split a node
              'min_samples_leaf': 100,          # the minimal sample required to be in leaf node
              'min_weight_fraction_leaf': 0.05, # the fraction of samples required to be in the leaf
              'max_leaf_nodes': None,           # the maximal number of leaf nodes to be grown
              'min_impurity_split': 1e-7,       # the threshold of impurity for early stopping
              'class_weight': 'balanced',       # the weight to balance target classes
              'random_state': None,             # random seed
              'bootstrap'   : True,             # indicates if using boostrap sample
              'oob_score'   : True,             # indicates if using out of bag
                                                # sample
                                                # out of bag estimation
              'verbose'     : 0,                # verbosity on training
              'warm_start'  : False,            # indicates if using the previous
                                                # result to for new samples    
              'n_jobs'      : 2                 # how many parallel jobs             
            }
    sig = inspect.signature(ExtraTreesClassifier)
    params.update({k: v for k, v in kwargs.items() if k in sig.parameters})
    estimator = ExtraTreesClassifier(**params)
    return estimator


def construct_boost_ensemble(**kwargs):
    """
    create an gradient-based ensemble 
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
    return estimator


def construct_dumb_classifier(**kwargs):
    """
    function to return a DummyClassifier
    """
    sig = inspect.signature(DummyProxy)
    params = {k: v for k, v in kwargs.items() if k in sig.parameters}
    estimator = Pipeline([('classifier', DummyProxy(**params))])
    return estimator


def construct_discriminant(**kwargs):
    """
    construct a linear discriminant for dimensionality reduction
    """
    params = {'solver': 'eigen',        # solver to compute convariance matrix, options are
                                        # svd (default, large data)
                                        # lsqr, least square solution can combine shrinkage 
                                        # eigen, can combine shrinkage  
              'shrinkage': None,        # set shrinkage for lsqr and eigne solver, options are
                                        # None (default, no shrinkage for svd), 
                                        # 'auto': apply shrinkage based on Ledoit-Wolf lemma
                                        # 'float': any float number within [0, 1]
              'priors': None,           # class pirors
              'n_components': 150,      # number of dimensionality for reduction
              'store_covariance': True, # compute the covariance matrix per-class
              'tol': None,              # tolerance for svd solver
             }

    sig = inspect.signature(LinearDiscriminantAnalysis)
    params.update({k: v for k, v in kwargs.items() if k in sig.parameters})
    estimator = LinearDiscriminantAnalysis(**params)
    return estimator


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
                'presort'     : 'auto',           # indicates if using sorting
                                                  # to speed up
                'min_impurity_split': 1e-7        # the threshold of impurity for early stopping
               }
    tree_kws.update(kwargs)
    tree_ba = inspect_and_bind(DecisionTreeClassifier, **tree_kws)
    estimator = DecisionTreeClassifier(*tree_ba.args, **tree_ba.kwargs)
    return estimator


def construct_naive_bayes(**kwargs):
    """
    providing default construction for naive bayes
    """
    steps = deque()
    for k, v in kwargs.items():
        if isinstance(v, BaseEstimator):
            if hasattr(v, 'predict'): # classifier
                steps.append((k, v))
            else:
                steps.appendleft((k, v))
    if len(steps) > 1:
        estimator = Pipeline(list(steps))
    else:
        assert('classifier' in kwargs)
        estimator = kwargs['classifier']
    return estimator


def construct_adaboost(base_estimator, **kwargs):
    """
    Parameters:
    -----------
    base_estimator: string
        the class name of base estimator used within AdaBoost
    """
    params = {
        'base_estimator': 
            predictor_construct[base_estimator](**kwargs), # default is DecisionTre
        'n_estimators': 10,                                # default is 50
        'learning_rate': 1.0,                              # deafult is 1.0
        'algorithm': 'SAMME.R',                            # default is SAMME.R, 
                                                           # requires predict_proba in
                                                           # base_estimator 
        'random_state': None,
    }
    if isinstance(params['base_estimator'], Pipeline):
        # only need classifier
        params['base_estimator'] = params['base_estimator'].steps[-1][-1]
    params.update(kwargs)
    param_ba = inspect_and_bind(AdaBoostClassifier, **params)
    estimator = AdaBoostClassifier(*param_ba.args, **param_ba.kwargs)
    return estimator


def construct_ovr(base_estimator, **kwargs):
    """
    Parameters:
    -----------
    base_estimator: string
        the class name of base estimator used within AdaBoost
    """
    steps = []
    base_estimator = predictor_construct[base_estimator](**kwargs)
    if isinstance(base_estimator, Pipeline):
        # only need classifier
        steps = [step for step in base_estimator.steps[:-1]]
        base_estimator = base_estimator.steps[-1][-1]
    
    params = {
        'base_estimator': base_estimator,
        'method': 'sigmoid',                                        # sigmoid or isotonic
        'cv' : StratifiedShuffleSplit(n_splits=1, test_size=0.2),   # reserving additional one as fitting validation  
    }

    params.update(kwargs)
    param_ba = inspect_and_bind(CalibratedClassifierCV, **params)
    estimator = Pipeline(steps + 
            [('classifier', CalibratedClassifierCV(*param_ba.args, **param_ba.kwargs))])
    return estimator


def construct_calibrator(n_classes, **kwargs):
    params = {
        'base_estimator': None,                                     # for global CalibratedClassifierC
        'method': 'sigmoid',                                        # sigmoid or isotonic
        'classes': numpy.arange(n_classes),                         # 
        'normalized': False,                                        # return normalized probability
    }
    params.update(kwargs)
    param_ba = inspect_and_bind(CalibratedClassifier, **params)
    return param_ba


def construct_scaler(base_estimator, preproc=None, **kwargs):
    estimator = predictor_construct[base_estimator](**kwargs)
    # update preproc
    return estimator


    """
    """


def train_test_split(n_samples, **kwargs):
    """
    function to make train or test split by random or by the predefine split

    Parameters
    ----------
    @param n_samples: int
        used to specify how many samples are used to create train and test set
        if not all used 
    @param kwargs: dict
        keywords used in sklearn.cross_validation.ShuffleSplit if applicable
    """
    predefine = [('n_train', 8544), ('n_dev', 1101), ('n_test', 2210)]
    
    if n_samples == sum(map(operator.itemgetter(1), predefine)):
        logging.info("using predefine split #train={n_train} #valid={n_dev} "
                "#test={n_test}".format(**dict(predefine)))
        sizes = numpy.asarray(list(accumulate(map(operator.itemgetter(1), predefine),
            operator.add)))
        assert(n_samples == sizes[-1])
        ttl_idx = numpy.arange(sizes[-1])
        return (ttl_idx[:sizes[0]], ttl_idx[sizes[0]:sizes[1]], 
                ttl_idx[sizes[1]:sizes[-1]])
    else: 
        # has no sufficient training examples (partially)
        if(not "test_size" in kwargs):
            kwargs["test_size"] = 0.15
        if(not "n_splits" in kwargs):
            kwargs["n_splits"] = 1
        indices = None 
        ttl_idx = numpy.arange(n_samples)
        splitter = ShuffleSplit(**kwargs)
        for split_idx, test_idx in splitter.split(ttl_idx):
            test = ttl_idx[test_idx]
            split_ = ttl_idx[split_idx]

        for train, valid in splitter.split(split_):
            indices = (split_[train], split_[valid], ttl_idx[test])

        assert(not set(indices[0]) & set(indices[1]))
        assert(not set(indices[0]) & set(indices[-1]))
        assert(not set(indices[-1]) & set(indices[1]))

        logging.info("using random split #train={n_train} #valid={n_dev} "
                "#test={n_test}".format(n_train=len(indices[0]),
                n_dev=len(indices[1]), n_test=len(indices[-1])))
        return indices


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
        

def preload_helper(csv_file, pretrain_loc=None, n_rows=-1, **kwargs):
    """
    load pretrain features and doing train-test split 

    Parameters
    ----------
    @param csv_file: string
        file path for extracted phrase information from parsing tree
    @param pretrain_loc: string
        file path for loading pretrained word embedding
    @param n_rows: int
        how many rows will be read in for training. will read all the rows 
        if the value is -1
    @param kwargs: extra keywords passing to train_test_split only works if 
        n_rows != -1 (parailly read in)
    """
    n_components, n_words = -1, -1
    if pretrain_loc:
        patch_pickled_preproc(pretrain_loc)
    else:
        # don't need pretrain word embedding for example, using single classifier 
        vocab, preproc = None, None
    n_words = len(vocab)
    features = transform_features(os.path.join(data_dir, csv_file), n_rows=n_rows, preproc=preproc,
                vocab=vocab)

    n_samples = len(features['wordtokens'])
    targets = numpy.hstack(list(map(lambda x: x.data[x.mask], 
        features['sentiments']))) # taking out root label

    # construct train / test split
    if n_rows != -1:
        train, valid, test = train_test_split(n_samples, random_state=n_rows, **kwargs)
    else:
        train, valid, test = train_test_split(n_samples, **kwargs)

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

    logging.info("preprocessor is loaded from %s using n_components=%d n_words=%d" %
            (pretrain_loc, n_components, n_words))

    return preproc, (train_features, targets[train]), \
        (valid_features, targets[valid]), (test_features, targets[test])


def construct_weight_model(preproc, predictor_maker, max_level, cvkwargs, gridkwargs, 
    strategy, global_propc=False, n_classes=5, **modelkwargs):

    """
    providing default construction for the label predictor mainly for weights
    selection in cross-validation settings

    Parameters
    ----------
    @param preproc: scikit-learn BaseEstimator or TransformMixin instance
        instance used to transform text features into numerical ones
    @param predictor_maker: string
         used to construct non-root sentiments predictors. avaiable options are "gnb" (GaussianNB),
         "dumb" (DummyClassifier) and "mnb" (MultiNomialNB, working on)
    @param max_level: int
         the maximal level used to construct non-root sentiments predictors
    @param cvkwargs: dict
        additional keywords will be passed into scikit-learn cross-validaiton
        instance
    @param gridkwargs: dict
        additional keywords will be passed into scikit-learn GridSearchCV
        instance 
    @param strategy: string
        indicate what probability calibration strategy should be taken. available options are 
        "calibrated" or "ovr" (global CalibratedClassifierCV calibrate the predict_proba output of AdaBoost)
        "ada" (local CalibratedClassifierCV calibrate the predict_proba output of AdaBoost's base estimator)
        None without any proability calibration
    @param global_propc: bool
        indicate if using a global preprocessor. doens't take effect in this function
    @param n_classes: int
        number of sentiment categories used for classification work (only accepts 2 or 5)
    @param modelkwargs: other keywords parameters passed to construct label predictor
  
    """    
    modelkwargs['weight_init'] = None
    if 'random_state' in modelkwargs:
        seed = modelkwargs.pop('random_state', None)

    # split based on the y-labels without considering sentence ids
    no_sentid_cv = StratifiedKFold(**cvkwargs)
    # split based on id 
    groupcvkws = cvkwargs.copy()
    if 'n_splits' in groupcvkws:
        groupcvkws['test_size'] = 1 / groupcvkws['n_splits']
    if 'shuffle' in groupcvkws:
        groupcvkws.pop('shuffle', None)

    sentid_cv = GroupShuffleSplit(**groupcvkws)  # don't use GroupKFold

    # adding calibrator
    wrappedkws, params = {}, None
    if strategy in ['calibrated', 'ovr']:
        params = {'base_estimator__weight_init': [None, 
                partial(compute_weights, 'weight_by_size'), # the shorter sentence the higher weight
                partial(compute_weights, 'weight_by_node')], # the higher level the lower weight   
              }
        control_searcher = GridSearchCV(estimator=CalibratedClassifierCV(
                                            base_estimator=predictor_maker(**modelkwargs),
                                            cv=StratifiedShuffleSplit(n_splits=1, test_size=0.2, 
                                                                      random_state=seed), 
                                            method='sigmoid'),
                                        param_grid=params,
                                        cv=no_sentid_cv, **gridkwargs)
    
        label_searcher = GridSearchCV(estimator=CalibratedClassifierCV(
                                            base_estimator=predictor_maker(**modelkwargs),
                                            cv=GroupShuffleSplit(n_splits=1, test_size=0.2, 
                                                                 random_state=seed),
                                            method='sigmoid'),
                                        param_grid=params, 
                                        cv=sentid_cv, **gridkwargs)
    elif strategy in ['ada']:
        params = {'weight_init': [None, 
                    partial(compute_weights, 'weight_by_size'), # the shorter sentence the higher weight
                    partial(compute_weights, 'weight_by_node')], # the higher level the lower weight   
                  }
        control_searcher = GridSearchCV(estimator=WeightAdaBoostClassifier(
                                            base_estimator=predictor_maker(
                                                    cv=StratifiedShuffleSplit(n_splits=1, test_size=0.2, 
                                                                              random_state=seed), 
                                                    **modelkwargs),
                                            weight_init=None), 
                                        param_grid=params, 
                                        cv=no_sentid_cv, **gridkwargs)
    
        label_searcher = GridSearchCV(estimator=WeightAdaBoostClassifier(
                                            base_estimator=predictor_maker(
                                                    cv=GroupShuffleSplit(n_splits=1, test_size=0.2,
                                                                         random_state=seed), 
                                                    **modelkwargs),
                                            weight_init=None),
                                        param_grid=params, 
                                        cv=sentid_cv, **gridkwargs)
    else:
        params = {'weight_init': [None, 
                    partial(compute_weights, 'weight_by_size'), # the shorter sentence the higher weight
                    partial(compute_weights, 'weight_by_node')], # the higher level the lower weight   
                  }
                   
        control_searcher = GridSearchCV(estimator=predictor_maker(**modelkwargs),
                                            param_grid=params, 
                                            scoring=make_scorer(construct_score),
                                            cv=no_sentid_cv, **gridkwargs)
    
        label_searcher = GridSearchCV(estimator=predictor_maker(**modelkwargs),
                                            param_grid=params, 
                                            scoring=make_scorer(construct_score),
                                            cv=sentid_cv, **gridkwargs)

    return label_searcher, control_searcher


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
        **kwargs)

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


def biased_prior_experiment(preproc, predictors, train_features, train_targets, cvkwargs, gridkwargs, 
    n_classes=5):
    """
    passing preset priors for experimenting label predictors
    """
    # ensure refit is not True also avoid inplace modification
    gridkwargs_pred = gridkwargs.copy()
    gridkwargs_pred['refit'] = False
    params = [
                (1/n_classes) * numpy.ones(n_classes),  # uniform
                *(numpy.asarray(arr)/numpy.sum(arr) for arr in [
                [0.8, 0.05, 0.05, 0.05, 0.05], # class 0 dominating
                [0.05, 0.8, 0.05, 0.05, 0.05], # class 1 dominating
                [0.05, 0.05, 0.8, 0.05, 0.05], # class 2 dominating
                [0.05, 0.05, 0.05, 0.8, 0.05], # class 3 dominating
                [0.05, 0.05, 0.05, 0.05, 0.8], # class 4 dominating
                ])
    ]
    search_maker = partial(GridSearchCV, param_grid={'classifier__priors': params})
    # filter unwanted kwargs
    sig = inspect.signature(search_maker, follow_wrapped=True)
    disjoint_keys = set(gridkwargs_pred.keys()) - set(sig.parameters.keys())
    for k in disjoint_keys:
        del gridkwargs_pred[k]

    # ensure cvkwargs uses the same seed
    cvkwargs['random_state'] = os.getpid()
    estimators = [search_maker(predictor, **gridkwargs_pred, cv=StratifiedKFold(**cvkwargs)) 
            for predictor in predictors] # include root level

    # checking params has correct name:
    if not 'classifier__priors' in estimators[0].get_params() and \
        'estimator__base_estimator__priors' in estimators[0].get_params():
        assert(len(estimators) == 1)
        estimators[0].param_grid['base_estimator__priors'] = \
        estimators[0].param_grid['classifier__priors']
        del estimators[0].param_grid['classifier__priors']

    logging.info("%d label estimators are constructed and start training ..." % max_level)
    
    levels = numpy.hstack(train_features['levels'])
    phrases = numpy.hstack(train_features['phrases']) # data type is object
    sentiments = numpy.hstack(train_features['sentiments'])
    estimators = group_fit(levels, phrases, sentiments, preproc, estimators, max_level)

    logging.info("complete %d label estimators ..." % max_level)

    return estimators


def construct_multiple_classifiers(predictor_maker, max_level, global_propc=None, **kwargs):
    """
    group phrases into levels and test on their accuracy with sentence only classifier to see 
        1. if there’s need to re-construct vocabulary set and truncated SVD 
        2. if there’s enough training examples for each level

    Parameters
    ----------
    @param preproc: sklearn.TransformMixin instance
        used to transform phrases into word vector based on BOW
    @param predictor_maker: sklearn.BaseEstimator instance
        class to constructor label_predictors for each level within max_level
    @param max_level: int
        the maximal integer used to construct label_predictors to ensure the training examples used
        sufficiently large
    @param kwargs: dict
        keyword arguments passing to predictor_maker
    """
    if global_propc:
        orig_steps = global_propc.steps
        orig_steps.extend([('normalizer', StandardScaler())]), # adding StandardScaler
        global_propc.set_params(steps = orig_steps)
    estimators = [predictor_maker(**kwargs) for _ in range(max_level)]
    return estimators


def multiple_naives(preproc, predictor_maker, vectorizer, max_level, 
    train_features, train_targets, cvkwargs, gridkwargs, strategy, 
    global_propc=False, n_classes=5, **modelkwargs):
    """
    construct multiple classifiers for each level 

    Parameters
    ----------
    @param preproc: scikit-learn BaseEstimator or TransformMixin instance
        instance used to transform text features into numerical ones
    @param predictor_maker: string
         used to construct non-root sentiments predictors. avaiable options are "gnb" (GaussianNB),
         "dumb" (DummyClassifier) and "mnb" (MultiNomialNB, working on)
    @param max_level: int
         the maximal level used to construct non-root sentiments predictors
    @param train_features: pandas.DataFrame
        store the training feaures for training set
    @param train_targets: numpy.array
        store the root sentiments for training set
    @param cvkwargs: dict
        additional keywords will be passed into scikit-learn cross-validaiton
        instance
    @param gridkwargs: dict
        additional keywords will be passed into scikit-learn GridSearchCV
        instance 
    @param strategy: string
        indicate what probability calibration strategy should be taken. available options are 
        "calibrated"  (global CalibratedClassifierCV calibrate the predict_proba output of label predictor)
        "ovr"  (local CalibratedClassifierCV calibrate the predict_proba output of label predictor)
        "biased" (doing biased expriment, no probability calibration)
        None (no calibration)
    @param global_propc: bool
        indicate if using a global preprocessor. doens't take effect in this function
    @param n_classes: int
        number of sentiment categories used for classification work (only accepts 2 or 5)
    @param modelkwargs: other keywords parameters passed to construct label predictor
    """
    normalized = modelkwargs.get('normalized')
    if global_propc:
        label_predictors = construct_multiple_classifiers(
            predictor_maker, max_level, global_propc=preproc, **modelkwargs)
    else:
         label_predictors = construct_multiple_classifiers(
            predictor_maker, max_level, global_propc=None, **modelkwargs)
    
    if strategy == 'biased':
        label_searchers = biased_prior_experiment(preproc, label_predictors,
            train_features, train_targets, cvkwargs, gridkwargs, n_classes=5)
        del label_predictors
    else:
        label_searchers = None

    transformers = []
    if label_searchers and hasattr(label_searchers[0], 'refit'):
        # iterate different label_predictors (pre-trained with different parameters)
        # this is a path predictor will go if its strategy is biased (including adaboost)
        for i, param in enumerate(label_searchers[0].cv_results_['params']):
            scores = numpy.asarray([est.cv_results_['mean_test_score'][i]
                for est in label_searchers])
            if 'classifier__priors' in param:
                param_value = param['classifier__priors']
            elif 'base_estimator__priors' in param:
                param_value = param['base_estimator__priors']
            else:
                param_value = ' '.join(list(param.values()))
            if hasattr(param_value, 'shape'):
                logging.info('label_predictors fit with priors=%s accuracy=%s' %(
                    numpy.array2string(param_value, suppress_small=True, precision=3),
                    numpy.array2string(scores, suppress_small=True, precision=3)))
            else:
                logging.info('label_predictors (# %d) fit with param=%s scores(%s)=%s' %(
                        len(label_searchers), param_value, label_searchers[0].scoring,
                        numpy.array2string(scores, suppress_small=True, precision=3)))
        label_predictors = [clone(est.estimator).set_params(**param) if not est.refit 
            else est.best_estimator_ for est in label_searchers]

    calibrator = None
    if strategy == 'calibrated':
        factory_args = construct_calibrator(n_classes, **modelkwargs)
    else:
        factory_args = None
    transformers.append(FunctionTransformer(
        labels_to_attributes(preproc, vectorizer, calibrator_args=factory_args), validate=False, 
        kw_args={'label_predictors': label_predictors}))
    return transformers, label_searchers


classifier_construct = {
    'tree': construct_decision_tree,
    'ensemble': construct_random_ensemble,
    'boost': construct_boost_ensemble,
}

predictor_construct = {
    'gnb'  : partial(construct_naive_bayes, classifier=GaussianNB()),
    'mnb'  : partial(construct_naive_bayes, classifier=MultinomialNB()),
    'mscale': partial(construct_scaler, normalizer=StandardScaler()),
    'gscale': construct_scaler,
    'dumb' : construct_dumb_classifier,
    'ada'  : construct_adaboost,
    'ovr'  : construct_ovr,
    'biased': None,
    'calibrated': None,
}


def recursive_construct(predictor_type):
    """
    recursive construct classifier based on the passing predictor_type pattern (split by underscore, _)
    
    Parameters
    ----------
    @param predictor_type: string
        string might have ([a-zA-Z]+_)*[a-zA-Z]+

    >>> res = recursive_construct("gnb")
    >>> print(res[0].func.__name__, res[1], res[2])
    construct_naive_bayes None None
    >>> res = recursive_construct("ada_gnb")
    >>> print(res[0].func.__name__, res[1], res[2])
    construct_adaboost ada None
    >>> res = recursive_construct("gscale")
    >>> print(res[0].__name__, res[1], res[2])
    construct_scaler None None
    >>> res = recursive_construct("ovr_gnb")
    >>> print(res[0].func.__name__, res[1], res[2])
    construct_ovr ovr None
    >>> res = recursive_construct("calibrated_gnb")
    >>> print(res[0].func.__name__, res[1], res[2])
    construct_naive_bayes calibrated calibrated
    >>> res = recursive_construct("calibrated_ada_gnb")
    >>> print(res[0].func.__name__, res[1], res[2])
    construct_adaboost ada calibrated
    """
    strategy_ = None
    if predictor_type in predictor_construct:
        predictor_maker = predictor_construct[predictor_type]
        clf_name =  None
    else:
        clf_name, base_clf = predictor_type.split('_', maxsplit=1)
        if base_clf.find('_') > 0:
            strategy_ = clf_name
            predictor_maker, clf_name, _ = recursive_construct(base_clf)
        else:
            if predictor_construct[clf_name]:
                predictor_maker = partial(predictor_construct[clf_name], base_clf)
            else:
                strategy_ = clf_name
                predictor_maker = predictor_construct[base_clf]
    
    """currently not consider handle partial(partial<>) recursive construct
    if strategy_ in predictor_construct:
        predictor_maker = partial(predictor_construct[strategy_ ], predictor_maker)
        strategy_ = None
    """
    return predictor_maker, clf_name, strategy_


def run_single_classifier(csv_file, classifier_type, cvkwargs, gridkwargs, batch_mode=False,
    n_rows=-1, presort=False, **kwargs):
    """
    training single classifier for root-sentiment classification problem where all the non-root
     sentiments are ground truth

    Parameters
    ----------
    @param csv_file: string
        file path for extracted phrase information from parsing trees
    @param classifier_type: string
        avaiable options are "tree" (DecisiontTreeClassifier), "ensemble" (ExtraTreeClassifier)
        and "boost" (GradientBoostClassifier)
    @param cvkwargs: dict
        keyword arguments to pass to construct cross-validation instance
    @param gridkwargs: dict
        keyword arguments to pass to construct GridSearchCV instance (not including cv)
    @param batch_mode: bool
        if True, function will return learning result (but not classifer); otherwise, dump the 
        learning result along with classifeirs 
    @param n_rows: int
        the number of rows will be read in; when n_rows = -1, will read in all the rows
    @param presort: bool
        if True, will sorting training examples based on their size
    @param kwargs: dict
        keyword arguments to pass to classifier construction
    """
   
    model_name = 'single.model'
    classifier_maker = classifier_construct[classifier_type]
    model_name = '%s_%s'%(classifier_type, model_name)

    preproc, (train_features, train_targets), (valid_features, valid_targets), \
        (test_features, test_targets) = preload_helper(csv_file, n_rows=n_rows)

    # construct models
    vectorizer = DictVectorizer(dtype=numpy.float32, sparse=True)
    tree_clf = classifier_maker(**kwargs) 

    # vectorize features
    train_mats = process_joint_features((train_features['sentiments'].tolist(), train_features), 
        vectorizer=vectorizer)
    valid_mats = process_joint_features((valid_features['sentiments'].tolist(), valid_features),
        vectorizer=vectorizer)
    train_mats = scipy.sparse.vstack([train_mats, valid_mats])
    targets = numpy.hstack([train_targets, valid_targets])

    if presort:
        train = len(targets) - len(valid_targets)
        test_fold = -1 * numpy.ones(len(targets), dtype=numpy.int8)
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
        unsort_targets = targets.copy()
        targets[:train] = unsort_targets[:train][sort_by_size]
        targets[train:] = unsort_targets[train:]
        del unsort_targets
        gridkwargs.update({'cv': PredefinedSplit(test_fold)})
    else:
        gridkwargs.update({'cv': StratifiedKFold(**cvkwargs)})

    learning_res = cal_learning_curve(tree_clf, train_mats, targets,
            train_sizes=numpy.linspace(0.1, 1.0, 50),
            **gridkwargs)

    # refit with all training data and test
    test_mats = process_joint_features((test_features['sentiments'].tolist(), test_features), 
        vectorizer=vectorizer)
    assert(train_mats.shape[-1] == test_mats.shape[-1])
    tree_clf.fit(train_mats, targets)
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

    final_result = (learning_res, numpy.mean(predictions == test_targets))
    if batch_mode:
        return final_result

    joblib.dump((tree_clf, vectorizer, preproc, final_result, 
        (train_features, train_targets), 
        (valid_features, valid_targets), 
        (test_features, test_targets)), 
            os.path.join(data_dir, model_name))
    logging.info('dumping {}'.format(model_name))


def run_multi_classifiers(csv_file, classifier_type, predictor_type, max_level, 
    pretrain_loc, cvkwargs, gridkwargs, batch_mode=False, presort=False, n_rows=-1, 
    n_classes=5, **kwargs):
    """
    training two-tier classifier for root-sentiment classification problem where all the non-root
     sentiments are predicted by label_predictor specified by predictor_type

    Parameters
    ----------
    @param: csv_file: string
        file path for the extracted phrases and releveant information from parsing trees
    @param: classifier_type: string
        avaiable options are "tree" (DecisiontTreeClassifier), "ensemble" (ExtraTreeClassifier)
        and "boost" (GradientBoostClassifier)
    @param: predictor_type: string
        used to construct non-root sentiments predictors. avaiable options are "gnb" (GaussianNB),
         "dumb" (DummyClassifier) and "mnb" (MultiNomialNB, working on)
    @param: max_level: int
        the maximal level used to construct non-root sentiments predictors
    @param pretrain_loc: string
        file path for the pre-train word embedding
    @param cvkwargs: dict
        keyword arguments to pass to construct cross-validation instance
    @param gridkwargs: dict
        keyword arguments to pass to construct GridSearchCV instance (not including cv)
    @param batch_mode: bool
        if True, function will return learning result (but not classifer); otherwise, dump the 
        learning result along with classifeirs 
    @param presort: bool
        if True, will sorting training examples based on their size
    @param n_rows: int
        the number of rows will be read in; when n_rows = -1, will read in all the rows
    @param n_classes: int
        number of sentiment categories used for classification work (only accepts 2 or 5)
    @param kwargs: dict
        keyword arguments to pass to classifier construction
    """
    from sklearn.model_selection import _search
    from unittest.mock import patch 

    preproc, (train_features, train_targets), (valid_features, valid_targets), \
        (test_features, test_targets) = preload_helper(
            csv_file, pretrain_loc=pretrain_loc, n_rows=n_rows)

    model_name = 'multi.model'
    classifier_maker = classifier_construct[classifier_type]
    model_name = '%s_%s_%s'%(classifier_type, predictor_type, model_name)

    # construct the final model
    vectorizer = DictVectorizer(sparse=(classifier_type!='boost'))
    classifier = Pipeline([('transformer', None), ('classifier', None)])

    # calling function which will train n-independent label predictors
    predictor_maker, clf_name, strategy_ = recursive_construct(predictor_type)
    if clf_name =='ada' and max_level != 1:
        max_level = 1
    transformers, label_searchers = multiple_naives(preproc, predictor_maker, vectorizer, max_level,
            train_features, train_targets, cvkwargs, gridkwargs, strategy=strategy_, 
            global_propc=(clf_name=='gscale'), n_classes=n_classes, **kwargs)

    gridkwargs.update({'cv': StratifiedKFold(**cvkwargs)})      
    
    # tuning the ensemble of classifiers and then refit with all training data
    with patch.object(_search, '_fit_and_score',  new_callable=_fit_and_score):
        params_grid = {'transformer': transformers, 'classifier': [classifier_maker(**kwargs)]}
        searcher = _search.GridSearchCV(classifier, params_grid, **gridkwargs)
        searcher.fit(pandas.concat([train_features, valid_features]), 
            numpy.hstack([train_targets, valid_targets]))
    test_score = searcher.score(test_features, test_targets)
    logging.info('completing fitting and test on test set with {} accuracy'.format(
                test_score))
    final_result = (searcher, label_searchers)
    if batch_mode:
        return (final_result, test_score)

    #import dill
    #with patch.multiple(joblib.numpy_pickle, pickle=dill, 
    with patch.multiple(joblib.numpy_pickle, 
        NumpyPickler=LanguageModelPickler, spec_set=True):
        joblib.dump((searcher, vectorizer, preproc, label_searchers, \
            (train_features, train_targets), (valid_features, valid_targets), \
            (test_features, test_targets)), os.path.join(data_dir, model_name))
    logging.info('completing fitting and dumping {}'.format(model_name))


def run_stacking_classifiers(csv_file, classifier_type, predictor_type, max_level, 
    pretrain_loc, cvkwargs, gridkwargs, batch_mode=False, presort=False, n_rows=-1, 
    n_classes=5, **kwargs):
    """
    training two-tier classifier for root-sentiment classification problem but with 
    label embedding and a SGD classifier on the top
    """
    preproc, (train_features, train_targets), (valid_features, valid_targets), \
        (test_features, test_targets) = preload_helper(
            csv_file, pretrain_loc=pretrain_loc, n_rows=n_rows)

    model_name = 'multi.model'
    classifier_maker = classifier_construct[classifier_type]
    model_name = '%s_%s_%s'%(classifier_type, predictor_type, model_name)

    # construct the final model
    vectorizer = DictVectorizer(sparse=(classifier_type!='boost'))
    classifier = Pipeline(
        [('transformer', None),
         ('classifier', None)])

    # calling function which will train n-independent label predictors
    if predictor_type in predictor_construct:
        predictor_maker = predictor_construct[predictor_type]
        transformers, label_searchers = multiple_naives(preproc, predictor_maker, vectorizer, max_level, 
            train_features, train_targets, cvkwargs, gridkwargs, **kwargs)
    else:
        if max_level != 1:
            max_level = 1
        clf_name, base_clf = predictor_type.split('_')
        predictor_maker = partial(predictor_construct[clf_name], base_clf)
        transformers, label_searchers = multiple_naives(preproc, predictor_maker, vectorizer, max_level,
            train_features, train_targets, cvkwargs, gridkwargs, **kwargs)

    # creating label embedding
    gridkwargs.update({'cv': StratifiedKFold(**cvkwargs)})      
 


def run_sample_weight_cv(csv_file, predictor_type, max_level, pretrain_loc, 
    cvkwargs, gridkwargs, batch_mode=False, presort=False, n_rows=-1, 
    n_classes=5, **kwargs):
    """
    employ four boosted naive bayes with different sample weights

    @param: csv_file: string
        file path for the extracted phrases and releveant information from parsing trees
    @param: predictor_type: string
        used to construct non-root sentiments predictors. avaiable options are "gnb" (GaussianNB),
         "dumb" (DummyClassifier) and "mnb" (MultiNomialNB, working on)
    @param: max_level: int
        the maximal level used to construct non-root sentiments predictors
    @param pretrain_loc: string
        file path for the pre-train word embedding
    @param cvkwargs: dict
        keyword arguments to pass to construct cross-validation instance
    @param gridkwargs: dict
        keyword arguments to pass to construct GridSearchCV instance (not including cv)
    @param presort: bool
        if True, will sorting training examples based on their size
    @param n_rows: int
        the number of rows will be read in; when n_rows = -1, will read in all the rows
    @param n_classes: int
        number of sentiment categories used for classification work (only accepts 2 or 5)
    @param kwargs: dict
        keyword arguments to pass to classifier construction

    """
    preproc, (train_features, train_targets), (valid_features, valid_targets), \
        (test_features, test_targets) = preload_helper(
            csv_file, pretrain_loc=pretrain_loc, n_rows=n_rows)

    predictor_maker, clf_name, strategy_ = recursive_construct(predictor_type)
    if clf_name =='ada' and max_level != 1:
        max_level = 1
  
    label_searcher, control_searcher = construct_weight_model(preproc, predictor_maker, 
        max_level, cvkwargs, gridkwargs, strategy_, **kwargs)

    train_mats, train_sentiments = phrase_only_transformation(preproc, train_features)
    
    logging.info('start sentidcv + GridSearchCV fitting')
    start_time = time.time()
    label_searcher.fit(train_mats, train_sentiments, groups=train_mats['ids'])
    logging.info(
        'complete sentidcv + GridSearchCV fitting (#{} samples) in {:.2f} seconds'.format(
            len(train_mats), time.time() - start_time))
    param_names = list(chain.from_iterable(map(lambda x: list(x.keys()), 
                  label_searcher.cv_results_['params'])))
    best_index = label_searcher.cv_results_['rank_test_score'][0] - 1

    valid_mats, valid_sentiments = phrase_only_transformation(preproc, valid_features)
    logging.info('best estimator = {}, average validation score = {:.3f}, '
                 'test score = {:.3f} out of #{} samples'.format(
                  param_names[best_index], 
                  label_searcher.cv_results_['mean_test_score'][best_index],
                  label_searcher.best_estimator_.score(valid_mats, valid_sentiments), 
                  len(valid_mats)))

    logging.info('start randomcv + GridSearchCV fitting')
    start_time = time.time()
    control_searcher.fit(train_mats, train_sentiments)
    logging.info(
        'complete sentidcv + GridSearchCV fitting (#{} samples) in {:.2f} seconds'.format(
        len(train_mats), time.time() - start_time))
    logging.info('no weighting control, average validation score = {:.3f} '
                 'test score = {:.3f} out of #{} samples'.format(
                 control_searcher.cv_results_['mean_test_score'][0],
                 control_searcher.best_estimator_.score(valid_mats, valid_sentiments),
                 len(valid_mats)))
    
    # dump classifiers
    fn = pathlib.Path(data_dir).joinpath("weight_%s.model" % (predictor_type))
    joblib.dump((label_searcher, control_searcher), fn)
    logging.info('store {}'.format(str(fn)))


def configure(config_type):
    """
    read exec.ini file for extra configuration
    """
    default_config = {
    'single': [{'n_splits': 10, 'random_state': None}, # used to create the same partition
               {'scoring': 'accuracy', 'verbose': 0},
               {}, # passing label_predictors configurations
               ],
    'multi':  [{'n_splits': 10, 'random_state': None}, # used to create the same partition
               {'scoring': 'accuracy', 'verbose': 0, 'n_jobs': 2, 'refit': True},
               {}, # passing label_predictors configurations
               ],
    'weight': [{'n_splits': 10, 'shuffle': True, 'random_state': None},
               {'verbose': 10, 'refit': True},
               {}, # passing label_predictors configurations
               ]
    }
    cvkws, gridkws, modelkws = default_config[config_type]
    config_file = 'exec.json'
    if os.path.exists(config_file):
        with open(config_file, 'rt') as fp:
            site_config = json.load(fp)
        if config_type in site_config:
            cvkws.update(site_config[config_type].get('cv', {}))
            gridkws.update(site_config[config_type].get('grid', {}))
            modelkws.update(site_config[config_type].get('model', {}))
    return cvkws, gridkws, modelkws


avaiables = ['gnb', 'dumb', 
             'ada_gnb',  # adaboost with gnb
             'ovr_gnb',  # ovr gnb with local calibrators
             'gscale_gnb', 'mscale_gnb', 
             'calibrated_gnb', 'calibrated_ada_gnb', # global calibrator
             'ada_ovr_gnb' # local calibrator (calibrator is built on top of gnb not ada)
             ]


def main(*args):
    parser = argparse.ArgumentParser(
        description="construct different models and execute experiments")
    parser.add_argument("subcommand", 
        choices=['weight', 'single', 'multi', 'preprocess'],
            help="[weight_training|single_classifier|multi_classifier|preprocess_features]")
    parser.add_argument("-f", "--file", default='treebased_phrases.csv', help="csv file for data")
    parser.add_argument("-n", "--nrows", type=int, default=-1, help="number of rows to read in")
    parser.add_argument("-w", "--nwords", type=int, default=4500, help="size of vocabulary")
    parser.add_argument("-c", "--ncomponents", type=int, default=300, help="maximal number of components used")
    parser.add_argument("-t", "--max_levels", type=int, default=16, help="threadshold for maximal level used")
    parser.add_argument("--presort", action="store_true", help="presorting training examples")
    parser.add_argument("--normalized", action="store_true", help="normalize predicting probability by simply dividing sum")
    parser.add_argument("--pretrain", default='treebased_phrases_vocab.model', 
        help="loading pretrained word-embeddings")
    parser.add_argument("--classifier", default="tree", choices=['tree', 'ensemble', 'boost'],
        help="specifying the type of top classifier, options are [tree|ensemble|boost]")
    parser.add_argument("--predictor", default="gnb", choices=avaiables,
        help="specifying the type of label predictors, options are [%s]" % ('|'.join(avaiables)))
    parser.add_argument("--seed", type=int, nargs=2, default=None, 
        help="specify the seed used for data splitting and modeling")
    args = parser.parse_args()
    data_seed = args.seed[0]
    if len(args.seed) > 1:
        model_seed = args.seed[1]
    else:
        model_seed = None
    if args.subcommand.startswith('w'):
        logging.info("start weighted cross-validation")
        cvkws, gridkws, modelkws = configure('weight')
        cvkws.update({'random_state': data_seed}) # for creating consistent split
        run_sample_weight_cv(os.path.join(data_dir, args.file), 
            args.predictor, args.max_levels, args.pretrain, 
            cvkws, gridkws, presort=args.presort, n_rows=args.nrows, 
            random_state=model_seed,**modelkws)
    elif args.subcommand.startswith('s'):
        logging.info("start using decision tree to predict sentiment"
                " file: %s", args.file)
        cvkws, gridkws, modelkws = configure('single')
        cvkws.update({'random_state': data_seed}) # for creating consistent split
        run_single_classifier(os.path.join(data_dir, args.file), args.classifier,
                cvkws, gridkws, presort=args.presort, n_rows=args.nrows, 
                random_state=model_seed, **modelkws) # for creating randomness in model
    elif args.subcommand.startswith('m'):
        logging.info("start using naive bayes + decision tree to predict"
                " sentiment file: %s", args.file)
        cvkws, gridkws, modelkws = configure('multi')
        cvkws.update({'random_state': data_seed}) # for creating consistent split
        run_multi_classifiers(args.file, args.classifier, 
                args.predictor, args.max_levels, args.pretrain, 
                cvkws, gridkws, presort=args.presort, n_rows=args.nrows, 
                random_state=model_seed, **modelkws)  # for creating randomness in model
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