from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.metrics import log_loss, pairwise
from sklearn.ensemble import VotingClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.validation import NotFittedError
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import BernoulliNB
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import Normalizer, normalize, MaxAbsScaler
from sklearn.preprocessing import OneHotEncoder 
from sklearn.feature_extraction.text import (TfidfVectorizer, strip_accents_unicode, 
                                             strip_accents_ascii)
from sklearn.preprocessing import Imputer, LabelEncoder
from sklearn.preprocessing import PolynomialFeatures 
from sklearn.kernel_approximation import (Nystroem, RBFSampler, AdditiveChi2Sampler, 
                                          SkewedChi2Sampler)
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomTreesEmbedding
from sklearn.externals.six import iteritems
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import validation_curve
from sklearn.manifold import SpectralEmbedding
from sklearn.svm import SVC
from sklearn.model_selection import _search as grid_search
from sklearn.feature_selection import SelectFdr, chi2
from operator import itemgetter
from collections import OrderedDict, defaultdict, Iterable, Counter
from functools import wraps, update_wrapper, partial
from nltk.tokenize.regexp import regexp_tokenize
from wordcloud import WordCloud
from enum import Enum
from scripts.run_naive_bayes import construct_decision_tree
from scripts.run_naive_bayes import cal_learning_curve 

from symlearn.utils.estimator import ItemSelector, FeatureCombiner
from symlearn.utils import curve_utils, JointScorer

from hyperopt import hp
from hyperopt.mongoexp import MongoTrials
from io import StringIO
from itertools import groupby

import contextlib
import scipy as sp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import traceback
import operator
import time
import numpy
import re
import os
import inspect
import sys
import multiprocessing
import datetime
import hyperopt.pyll.base as pyll
import logging

logger = logging.getLogger(__name__)

def wrap_fit_and_score(train_scores):

    @wraps(grid_search._fit_and_score)
    def collect_train_score(*args, **kwargs):
        nonlocal train_scores
        new_kws = {'return_train_score': True}
        new_kws.update(kwargs)
        score_res = _fit_and_score(*args, **new_kws)
        train_scores.append(score_res[0])
        return(score_res[1:])
    return(collect_train_score)

def remap_naviebayse_proba(bayes, true_y):
  predictor = bayes
  expect = true_y
  def get_prob(X, y=None):
    nonlocal predictor
    nonlocal expect
    try:
      check_is_fitted(predictor, ["coef_"])
    except NotFittedError:
      predictor = predictor.fit(X, expect)
    proba = predictor.predict_proba(X)
    return(proba)
  return(get_prob)


def combine_animal_type(X, animal_types):
    """
    allow generate features with specific group codes

    Parameters
    ==========
    X: numpy or scipy sparse matrix of n_samples x n_features dimension which
    is used to be combined with interactor based on the animal_types  
    animal_types: numpy or scipy sparse matrix of n_samples x n_group dimension
    in one-hot-encoding for the animal types
    """
    # if both are sparse array, then return sparse array
    if isinstance(X, sp.sparse.spmatrix) and \
            isinstance(animal_types, sp.sparse.spmatrix):
        return(sp.sparse.hstack([animal_types[:, i].multiply(X) 
            for i in range(animal_types.shape[1])], format=X.format,
            dtype=X.dtype))
    else:
        # otherwise, return dense
        if hasattr(X, 'todense'):
            X = X.todense()
        if hasattr(animal_types, 'todense'):
            anima_types = animal_types.todense()
        return(numpy.hstack([animal_types[:, i] * X
            for i in range(animal_types.shape[1])]))


class combine_temporal_info:
    """
    an experimental function object for converting DateTime information into
    required format 
    """
    def __init__(self, **kwargs):
        self.return_raw = kwargs.get('return_raw', False)
        self.return_histogram = kwargs.get('return_histogram', False)
        self.as_sorted = kwargs.get('as_sorted', True)

    def cal_density(self, fn, X, y, sample_weights = None):
        ttl_counts, bins = fn(X, bins='doane', density=True,
                range=(0, 366))
        bin_width = bins[1] - bins[0]
        if not sample_weights is None:
            wts = Counter(sample_weights)

        @wraps(fn)
        def histogram(X, sample_weights=None):
            bin_index = np.floor(X / bin_width).astype(np.int)
            density = ttl_counts[bin_index]
            if sample_weights is None:
               sample_wts = numpy.ones((len(X),))
            else:
               assert(not wts is None)
               sample_wts = np.asarray([wts[s] for s in sample_weights])
            density *= sample_wts
            density = normalize(density.reshape(-1, 1), norm=self.norm, axis=0) 
            return(density, bin_index)

        return(histogram)

    def count_yearly_days(self, X, base_scale='h'):
        dtype = re.sub(r"\[.*\]", '', str(X.dtype))
        days = X.astype(("%s[%c]" % (dtype, base_scale)))
        years = X.astype(("%s[Y]" % dtype))
        return((days - years).astype(np.int), years.astype(np.str))

    def __call__(self, X, y=None, **fit_params):
        assert(isinstance(X, np.ndarray))

        # using DateTime information as categorical features
        if self.return_raw:
           days, years = self.count_yearly_days(X, base_scale='D')
           return([dict([('YearlyEventDays', da), ('Year', ye)]) 
               for da, ye in zip(days, years)]) 
        
        if self.return_histogram:
            # compute histogram  
            if histogram is None:
               histogram = self.cal_density(np.histogram, days, y, sample_weights=years)

            density, bin_index = histogram(days, sample_weights=years)
            if density.ndim == 1 or density.shape[1] != 1:
                density = density.reshape(-1, 1)
            return(np.hstack(bin_index, density))

        if self.as_sorted:
            base_units, years = self.count_yearly_days(X, base_scale='h')
            if fit_params.get('with_year', False):
                return(numpy.asarray(sorted([(int(ye), da, i) for i, (da, ye)
                    in enumerate(zip(base_units, years))], key=itemgetter(0, 1)))) 
            else:
                return(numpy.asarray(sorted([(da, i) for i, (da, ye) in
                    enumerate(zip(base_units, years))], key=itemgetter(0, 1)))) 

class generlized_additivechi2sampler:
    """
    an experimental function object to conduct time sampling given time span
    instead of fixed sample proportion in original AdditiveChi2Sampler
    """
    def __init__(self, sampler=None, proc_method='mean'):
        if sampler is None:
            self.sampler = AdditiveChi2Sampler()
        else:
            self.sampler = sampler
        self.proc_method = proc_method

    def __call__(self, X, temporal_info, y=None):
        # transform from n_samples to n_timepoint
        key = numpy.arange(temporal_info.shape[1] - 1).tolist()
        tp_query = OrderedDict([(timepoint, list(map(itemgetter(-1), group)))
            for timepoint, group in groupby(temporal_info, itemgetter(*key))])
        tp2idx = dict([(tp, i) for i , tp in enumerate(tp_query.keys())]) 

        if isinstance(X, sp.sparse.spmatrix):
            Xt = sp.sparse.lil_matrix((len(tp_query), X.shape[1]), dtype=X.dtype)
        else:
            Xt = numpy.zeros((len(tp_query), X.shape[1]), dtype=X.dtype)
        
        weights = numpy.empty(len(tp_query), dtype=numpy.object) 
        for timepoint, inds_group in tp_query.items(): 
            group_x = X[tuple(inds_group), :] 
            assert(hasattr(group_x, self.proc_method))
            proc_func = getattr(group_x, self.proc_method)
            centroid = proc_func(axis=0)
            if isinstance(centroid, np.matrix):
                centroid = centroid.A1
            Xt[tp2idx[timepoint]] = centroid 
            weights[tp2idx[timepoint]] = pairwise.cosine_similarity(
                    group_x, centroid.reshape(1, -1))
            assert(numpy.all(weights[tp2idx[timepoint]] > 0))

        # pass to sampler to augment features
        if isinstance(X, sp.sparse.spmatrix):
            Xt = type(X)(Xt)
        Xt = self.sampler.fit_transform(Xt)

        # convert back to n_samples  
        if isinstance(X, sp.sparse.spmatrix):
            Xs = sp.sparse.lil_matrix((X.shape[0], Xt.shape[1]), dtype=X.dtype)
        else:
            Xs = numpy.zeros((X.shape[0], Xt.shape[1]), dtype=X.dtype)
        for timepoint, inds_group in tp_query.items():
            Xs[tuple(inds_group), :] = weights[tp2idx[timepoint]] * Xt[tp2idx[timepoint]]
        if isinstance(X, sp.sparse.spmatrix):
            Xs = type(X)(Xs)
        return(Xs)

def build_preprocess(phrase, func, **kwargs):
    replace_dict = {'retr' : 'retriever',
                    'terr': 'terrier',
                    'wirehaired': 'wirehair',
                    'coated': 'coat'}
    words = func(phrase, **kwargs)
    return([replace_dict[w] if w in replace_dict else w for w in words])
  

def vocabs2index(vocabs):
  index2vocabs = numpy.empty((len(vocabs),), dtype=training_data['Breed'].dtype)
  for k, v in vocabs.items():
    index2vocabs[v] = k
  return(index2vocabs)
    

def show_cluster_result(vocabs, X, spec_cluster):
  index2vocabs = vocabs2index(vocabs)
  n_clusters = spec_cluster.n_clusters
  bicluster_ncuts = list(bicluster_ncut(X, i, spec_cluster)
                         for i in range(n_clusters))
  best_idx = np.argsort(bicluster_ncuts)[:n_clusters]

  print()
  print("Best biclusters:")
  print("----------------")
  for idx, cluster in enumerate(best_idx):
      n_rows, n_cols = spec_cluster.get_shape(cluster)
      cluster_docs, cluster_words = spec_cluster.get_indices(cluster)
      if not len(cluster_docs) or not len(cluster_words):
          continue

      # categories
      counter = defaultdict(int)
      for i in cluster_docs:
          counter[training_data['OutcomeType'].iloc[i]] += 1
      cat_string = ", ".join("{:.0f}% {}".format(float(c) / n_rows * 100, name)
                             for name, c in most_common(counter))

      # words
      out_of_cluster_docs = spec_cluster.row_labels_ != cluster
      out_of_cluster_docs = np.where(out_of_cluster_docs)[0]
      word_col = X[:, cluster_words]
      word_scores = np.array(word_col[cluster_docs, :].sum(axis=0) -
                             word_col[out_of_cluster_docs, :].sum(axis=0))
      word_scores = word_scores.ravel()
      kmost_common = (10 >= len(cluster_words)) * len(cluster_words) + \
                     (10 < len(cluster_words)) * 10
      important_words = list(index2vocabs[cluster_words[i]]
                             for i in word_scores.argsort()[:-10:-1])

      print("bicluster {} : {} documents, {} words".format(
          idx, n_rows, n_cols))
      print("categories   : {}".format(cat_string))
      print("words        : {}\n".format(', '.join(important_words)))


def bicluster_ncut(X, i, cluster):
    rows, cols = cluster.get_indices(i)
    # if have no members, then return the max values
    if not (np.any(rows) and np.any(cols)):
        return sys.float_info.max
    # finding the rows and cols are not cluster i
    row_complement = np.nonzero(np.logical_not(cluster.rows_[i]))[0]
    col_complement = np.nonzero(np.logical_not(cluster.columns_[i]))[0]
    # Note: the following is identical to X[rows[:, np.newaxis], cols].sum() but
    # much faster in scipy <= 0.16
    weight = X[rows][:, cols].sum() # getting total counts
    # samples are not in desired features / cols and features not in desired rows
    cut = (X[row_complement][:, cols].sum() +
           X[rows][:, col_complement].sum())
    return cut / weight # smaller better
  

def most_common(d):
    """Items of a defaultdict(int) with the highest values.

    Like Counter.most_common in Python >=2.7.
    """
    return sorted(iteritems(d), key=itemgetter(1), reverse=True)


def plot_confusion_matrix(cm, target_names, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    # plt.colorbar()
    tick_marks = np.arange(len(target_names))
    plt.xticks(tick_marks, target_names, rotation=300)
    plt.yticks(tick_marks, target_names)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# OutcomeSubType => half missing (17 subtypes including NaN)
# handle SexuponOutcome => consider to split the column into Male/Female vs Neutered / Spayed / Intact
def split_sexuponoutcome(val):
  if type(val) != str or "Unknown" == val:
    return(np.repeat("Unknown", 2))
  else: 
    return(val.split(' '))  


# AgeuponOutcome => consider to transform the column into years
def split_ageuponoutcome(val):
  unit_map = {'year': 1, 'month': 1/12, 'week': 1/52, 'day': 1/365}
  if type(val) != str:
    return(None)
  else: 
    num, units = val.split(' ')
    num = float(num)
    if units in unit_map:
      scale = unit_map[units]
    elif units[:-1] in unit_map:
      scale = unit_map[units[:-1]]
    return(scale * num)


def building_word_feature(**kwargs):
    preproc = partial(build_preprocess, func=regexp_tokenize, 
            pattern=r"(?u)\b\w\w+\b") # cannot pickle 
    features = kwargs.pop('features', ['Breed', 'Color'])
    assert(not 'YearlyEventDays' in features) 
    stacking = []
    for name in features:
        procedures = [
                ('selector', ItemSelector(name)),
                ('vectorizer', TfidfVectorizer(
                    binary=False, norm='l2', use_idf=True, smooth_idf=True, 
                    sublinear_tf=True, tokenizer=preproc, dtype=numpy.float64,
                    strip_accents='ascii', stop_words=['unknown'],
                    ngram_range=(1, 2)))]
        feature_name = "%s_features" %(name.lower())
        stacking.append((feature_name, Pipeline(steps=procedures)))

    weights = [(name, 1/len(stacking)) for name, _ in stacking]
    models = [('generator', FeatureCombiner(
        [('features', FeatureUnion(stacking, transformer_weights=dict(weights))),
         ('interactor', Pipeline(steps=[
             ('selector', ItemSelector('AnimalType')),
             ('vectorizer', TfidfVectorizer(
                    binary=True, norm='l2', use_idf=False, smooth_idf=False, 
                    sublinear_tf=False, tokenizer=preproc, dtype=numpy.float64,
                    strip_accents='ascii', stop_words=['unknown'],
                    ngram_range=(1, 1)))]))],
             combine_animal_type))]
    extractor = Pipeline(models) 
    return(extractor)

def construct_bow_classifier(**kwargs):
    n_jobs = kwargs.pop('n_jobs', 1)
    verbose = kwargs.get('verbose', 0)
    seed = kwargs.get('random_state', None)
    steps = []

    extractor = building_word_feature(**kwargs)

    if kwargs.pop("sample_time", True):
        embedder = FeatureCombiner([
            ('extractor', extractor), 
            ("time_features", Pipeline(steps=[
                ('selector', ItemSelector('DateTime')),
                ('vectorizer', FunctionTransformer(
                    combine_temporal_info(), validate=False, pass_y=True))]))],
                generlized_additivechi2sampler(sampler=AdditiveChi2Sampler(sample_steps=2)))
        steps.extend([('embedder' , embedder), ('scaler', MaxAbsScaler())])
    else:
        embedder = Pipeline(steps=[
            ('kernel', AdditiveChi2Sampler()),
            ('scaler', MaxAbsScaler())])
        steps.extend([('extractor', extractor), ('embedder', embedder)])
    if kwargs.pop("use_svc", True):
        classifier = SVC(C=1.0, kernel='linear', probability=True, verbose=True,
                decision_function_shape='ovo', max_iter=50, class_weight=None)
    else:
        classifier = SGDClassifier(loss='modified_huber', class_weight=None,
                n_iter=50) 
        if kwargs.pop('use_ovo', False):
            classifier = OneVsOneClassifier(classifier)
    steps.append(('classifier', classifier))

    clf = Pipeline(steps=steps)

    params = clf.get_params(deep=True)
    reset_keys = {s: verbose for s in params.keys() if s.endswith('verbose')}
    reset_keys.update({s: seed for s in params.keys() if s.endswith('random_state')})
    reset_keys.update({k: v for k, v in kwargs.items() if k in params})
    clf.set_params(**reset_keys)
    return(clf)

def construct_tree_classifier(**kwargs):
    select_features = kwargs.get('features', ['AnimalType', 'Sex', 'HasName',
                                 'ReproductiveStatus', 'LogAgeInYear'])
    n_jobs = kwargs.pop('n_jobs', 1)
    verbose = kwargs.get('verbose', 0)
    seed = kwargs.get('random_state', None)
    classifier = GradientBoostingClassifier(
            max_leaf_nodes=kwargs.get('max_leaf_nodes', 8),
            n_estimators=kwargs.get('n_estimators', 325),
            learning_rate=kwargs.get('learning_rate', 0.05), 
            subsample=kwargs.get('subsample', 0.7),
            loss='deviance')
    if kwargs.pop('use_binary', False):
        classifier = OneVsRestClassifier(classifier)
    features = [('main_features', Pipeline([('selector', ItemSelector(select_features)),
                                    ('vectorizer', DictVectorizer(sparse=False)),
                                    ('imputer', Imputer(missing_values='NaN', strategy='mean', axis=0)),
                                    ('generator', PolynomialFeatures(degree=2,
                                                  interaction_only=True,
                                                  include_bias=False))])),
                ('extra_features', Pipeline([('selector', ItemSelector('DateTime')),
                                     ('generator', FunctionTransformer(
                                         combine_temporal_info(return_raw=True),
                                         validate=False, pass_y=True)),
                                     ('vectorizer', DictVectorizer(sparse=False))]))]
    if kwargs.pop('use_word', False):
        extractor = building_word_feature(**kwargs)
        # need to screen vocabs and then add back AnimalType interactor
        raw_features = extractor.steps[0][-1].transformer_list[0]
        extractor.steps[0][-1].transformer_list.remove(raw_features)
        selector = Pipeline(steps=[raw_features,
            ('selector', SelectFdr(chi2, alpha=0.05))])
        extractor.steps[0][-1].transformer_list.insert(
                0, ('features', selector))
        extractor.steps.append(('denser', FunctionTransformer(lambda x:
            x.toarray(),  validate=True, accept_sparse=True)))
        features.append(('word_features', extractor))
    weights = {name: 1 / len(features) for name, _ in features}
    models = [('extractor', FeatureUnion(features, transformer_weights=weights)),
              ('classifier', classifier)]
    clf = Pipeline(models)

    params = clf.get_params(deep=True)
    reset_keys = {s: verbose for s in params.keys() if s.endswith('verbose')}
    reset_keys.update({s: seed for s in params.keys() if s.endswith('random_state')})
    clf.set_params(**reset_keys)
    return(clf)


def tune_training(training_data, searcher_name, params, **kwargs):
    n_fold = kwargs.pop('n_fold', 5)
    scoring = kwargs.pop('scoring', 'log_loss')
    verbosity = kwargs.get('verbosity', 10)
    n_jobs = kwargs.pop('n_jobs', 1)
    n_iter = kwargs.pop('n_iter', None)

    classifiers, data_gen = construct_model(training_data, **kwargs)
    X_train, X_test, test_inds = next(data_gen)
    y_train, y_test, _ = next(data_gen)
    y_train = y_train.values
    y_test = y_test.values
    results = {}
    for k, clf in classifiers.items():
        param_name, param_range = params
        formal_params = clf.get_params(deep=True)
        if not param_name in formal_params:
            logger.info('%s has no parameter %s' %(k, param_name))
            continue

        search_kw = {'cv': StratifiedKFold(y_train, n_fold, shuffle=False), 'n_jobs': n_jobs,
                     'scoring': JointScorer(scoring), 'verbose': verbosity}

        if searcher_name == 'hyperopt':
           searcher = default_hyperopt_setting(**kwargs) 
        elif searcher_name == 'validate':
            try:
               retval = validation_curve(clf, X_train, y_train,
                       *params,**search_kw)
            except:
               logger.warning(traceback.format_exc()) 
               continue
            results[k] = list()
            results[k].append(('train_scores', retval[0]))
            results[k].append(('test_scores', retval[1]))
            results[k].append(('parameters', [(param_name, param_range)]))
        else:
            if searcher_name == 'random':
                if n_iter is None:
                    n_iter = inspect.signature(
                            grid_search.RandomizedSearchCV).parameters['n_iter'].default
                candidates = [row.tolist() for row in param_range.rvs(n_iter * 10)]
                searcher = grid_search.RandomizedSearchCV(clf, 
                        {param_name: candidates}, **search_kw)
            elif searcher_name == 'grid':
                searcher = grid_search.GridSearchCV(clf, dict([params]), **search_kw)

            try:
                searcher = searcher.fit(X_train, y_train)
            except:
               logger.warning(traceback.format_exc()) 
               continue
            results[k] = list()
            results[k].append(('test_scores', 
                [gs.cv_validation_scores for gs in searcher.grid_scores_]))
            pa_lookup = defaultdict(list)
            for gs in searcher.grid_scores_:
                for k, v in gs.parameters.items():
                    pa_lookup[k].append(v)
            results[k].append(('parameters', pa_lookup))
        results['y_true'] = test_inds 
    return(classifiers, results)


def batch_training(training_data, **kwargs):
    n_fold = kwargs.pop('n_fold', 5)
    scoring = kwargs.pop('scoring', 'log_loss')
    verbose = kwargs.get('verbose', 10)
    n_jobs = kwargs.get('n_jobs', 1)

    samplesizes = kwargs.pop('samplesizes', numpy.linspace(0.5, 1.0, 5))

    classifiers, data_gen = construct_model(training_data, **kwargs)
    X_train, X_test, test_inds = next(data_gen)
    y_train, y_test, _ = next(data_gen)
    y_train = y_train.values
    y_test = y_test.values

    results = {}
    for k, clf in classifiers.items():
        results[k] = list()
        learning_res = cal_learning_curve(clf, X_train, y_train,
                train_sizes=samplesizes,
                learn_func=curve_utils.learning_curve,
                scoring=JointScorer(scoring), verbose=verbose,
                cv=StratifiedKFold(y_train, n_fold, shuffle=False),
                exploit_incremental_learning=False,
                n_jobs=n_jobs)
        results[k].append(('learning_curve', learning_res))

        clf = clf.fit(X_train, y_train)
        results[k].append(('train', clf.score(X_train, y_train)))
        results[k].append(('test', clf.score(X_test, y_test)))
        results[k].append(('y_pred', clf.predict(X_test)))
    results['y_true'] = test_inds 
    return(classifiers, results)


def proc_data(filename=os.path.join(os.environ['WORKSPACE'], 'Kaggle/WorkNote/ShelterAnimalOutcomeComp')):
    # parse_dates receiving 0-indexed 
    training_data = pd.read_csv(os.path.join(filename, "train.csv"), parse_dates=[2], infer_datetime_format=True, 
                                keep_date_col=True, na_values={'SexuponOutcome': 'Unknown'});

    # encode outcomeType using LabelEncoder (due to manually encode response
    # might result in errors in some estimators)
    if not hasattr(training_data['OutcomeType'], 'cat'):
      assert(numpy.all(training_data['OutcomeType'].isnull().values)==False)
      OutcomeType = Enum('OutcomeType', ' '.join(training_data['OutcomeType'].unique()), start=0)
      # convert to categorical type
      training_data['OutcomeType'] = training_data['OutcomeType'].astype('category')
      training_data['OutcomeType'].categories = list(OutcomeType.__members__)
      training_data['OutcomeCode'] =  training_data['OutcomeType'].apply(lambda x: OutcomeType[x].value)
      
    if not hasattr(training_data['OutcomeSubtype'].dtype, 'cat'): 
      def convert_to_subtype(val):
        if type(val) == str:
          return(OutcomeSubType[val.replace(' ', '_')].value)
        else:
          return(val)
  
      OutcomeSubType = Enum('OutcomeSubType', ','.join(
          map(lambda x: x.replace(" ", "_"), 
              training_data['OutcomeSubtype'][~training_data['OutcomeSubtype'].isnull()].unique())))
      training_data['SubOutcomeCode'] =  \
        training_data['OutcomeSubtype'].apply(convert_to_subtype)

      # convert to categorical type
      training_data['OutcomeSubtype'] = training_data['OutcomeSubtype'].astype('category') 
      training_data['OutcomeSubtype'].cat.categories = list(OutcomeSubType.__members__)
      
    split_result = np.asarray(list(map(split_sexuponoutcome, training_data['SexuponOutcome'].values)))
    training_data['Sex'] = split_result[:, 1]
    training_data['ReproductiveStatus'] = split_result[:, 0]

    training_data.loc[training_data['ReproductiveStatus'] == 'Spayed', 'ReproductiveStatus'] = 'Neutered'

    if not hasattr(training_data['Sex'].dtypes, 'cat'):
      training_data['Sex'] = training_data['Sex'].astype('category')
    if not hasattr(training_data['ReproductiveStatus'].dtypes, 'cat'):
      training_data['ReproductiveStatus'] = training_data['ReproductiveStatus'].astype('category')

    training_data['AgeInYear'] = training_data['AgeuponOutcome'].apply(split_ageuponoutcome)
    training_data['LogAgeInYear'] = training_data['AgeInYear'].apply(lambda x: np.max([np.log(1/8760), np.log(x)]))

    training_data['AnimalTypeCode'] = training_data['AnimalType'].apply(
      lambda x: 1 * (x == 'Dog') or -1 * (x == 'Cat'))
    training_data['HasName'] = ~training_data['Name'].isnull()
    training_data['ProcName'] =  training_data['Name'].fillna(value='Unknown')
    training_data['YearlyEventDays'] = training_data['DateTime'].apply(lambda
            d: (d - datetime.datetime(d.year, 1, 1)).days)
    training_data['AnimalType'] = training_data['AnimalType'].astype('category')
    training_data['AnimalType'].cat.categories = ['Dog', 'Cat']
    return(training_data)

def construct_model(training_data, **kwargs):
    all_features = training_data.columns.tolist()
    all_features.remove('OutcomeCode')

    X = training_data[all_features]
    y = training_data['OutcomeCode']

    classifiers = OrderedDict() 
    classifiers['vecfeat_extractor'] = construct_bow_classifier(**kwargs)
    classifiers['dictfeat_extractor'] = construct_tree_classifier(**kwargs)
    classifiers['classifier'] = VotingClassifier([('BOW_clf', classifiers['vecfeat_extractor']),
                                                  ('tree_clf', classifiers['dictfeat_extractor'])],
                                                  voting='soft',
                                                  weights=[0.2, 0.8])
    if kwargs.get('random_split', False):
        train, test = zip(*StratifiedShuffleSplit(y,
            random_state=kwargs.get('random_state', None), n_iter=1, test_size=0.2))
        return(classifiers, map(lambda s: (s.iloc[train], s.iloc[test], test), [X, y]))
    else:
        # picking 2014, 2015 examples as training; while 2013 and 2016 as test
        # data
        mask = training_data['DateTime'].apply(lambda x: (x.year == 2014 or
            x.year==2015)).values
        train = training_data.ix[mask, 'DateTime'].sort_values().index.values
        test = training_data.ix[
                numpy.logical_not(mask), 'DateTime'].sort_values().index.values
        return(classifiers, map(lambda s: (s.iloc[train], s.iloc[test], test), [X, y]))
          

def parse_training_output(func, bound_args, output_type):
    """
    Parameters
    ==========

    """
    capture_io = StringIO()
    with contextlib.redirect_stdout(capture_io):
        func(*bound_args.args, **bound_args.kwargs)
    output = capture_io.getvalue()
    if output_type == 'sgd':
        lines = [line.replace('\n', ', ').strip('., ') for line in
                output.split('--') if len(line)]
        pat = re.compile(r"(?:([a-zA-Z]+):? (-?\d+\.?\d*))")
        stats = [dict(re.findall(pat, line)) for line in lines] 
    capture_io.close()
    return(stats) 

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--tuning", 
        help="tuning training result with hyperopt", action="store")
    parser.add_argument("--sampling", 
        help="using small sample for testing", action="store_true")
    parser.add_argument("--partial", 
            help="parial train with incremental samples", action="store_true")
    parser.add_argument("--mongodb", 
            help="using mongodb for storing tuning result")
    args = parser.parse_args()
    training_data = proc_data()
    kwargs = {}
    if args.sampling is True: 
        select_ind = np.hstack([grpind[:2000] for _, grpind in
                                training_data.groupby(training_data['OutcomeCode']).indices.items()])
        select_ind.sort()
        sample_training = training_data.iloc[select_ind]
    else:
        sample_training = training_data
    if not args.tuning is None:
        tune_training(sample_training, args.tuning, connect_str=args.mongodb)
    else:
        classifiers, results = batch_training(sample_training, **vars(args))
