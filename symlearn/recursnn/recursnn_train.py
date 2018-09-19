# deprecated module, trash
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import fbeta_score, make_scorer, accuracy_score, precision_score
from sklearn.feature_selection import GenericUnivariateSelect, chi2, RFE
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.model_selection import (GridSearchCV, ParameterGrid,
        PredefinedSplit, StratifiedShuffleSplit, StratifiedKFold)
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import Normalizer
from sklearn.externals import joblib
from copy import deepcopy
from itertools import groupby
from operator import attrgetter,itemgetter
from collections import OrderedDict, deque

from symlearn.utils.wordproc import WordNormalizer, WordNormalizerWrapper
from . import recursnn_helper

import sklearn
import inspect
import logging
import nltk
import numpy as np
import pandas as pd
import random
import os

logger = logging.getLogger(__name__)

def split_train_parameters(searcher, parameter_iterable, X, y):
    """
    split training
    """

    searcher._fit(X, y, parameter_iterable)
    return(searcher)


def init_knockout(cv, seed=42, verbosity=10):
    """
    utitlity to generate parameters and grid search instance for knockout
    stage
    """

    # knock-out stage
    knockout_steps = [
        ('vectorizer', WordNormalizerWrapper(
            normalizer=WordNormalizer(
                norm_number=True, norm_punkt=True))),
        ('transformer', TfidfTransformer(sublinear_tf=True, use_idf=True)),
        ('selector', GenericUnivariateSelect(
            score_func=chi2, param=0.6, mode='fpr')), ('reducer', RFE(
                None, step=1, verbose=verbosity))]

    knockout_piper = Pipeline(steps=knockout_steps)
    knockout_tuner = GridSearchCV(
        knockout_piper,
        param_grid={},
        verbose=verbosity,
        n_jobs=-1,
        cv=cv)
    return(knockout_tuner)


def init_final(knockout_tuner, cv, verbosity):
    """
    utitlity to generate parameters and grid search instance for
    final train and predict stage
    """

    # only when using MultiNomial Naive Bayes, normalizer = SoftMaxScaler
    train_steps = [
        ('decomposer', TruncatedSVD()), ('normalizer', Normalizer),
        ('classifier', OneVsRestClassifier(None))]

    final_piper = Pipeline(steps=train_steps)
    final_tuner = GridSearchCV(
        final_piper,
        param_grid={},
        verbose=verbosity,
        n_jobs=-1)
    return(final_tuner)


def param_generator(
        stage=[
            'knockout',
            'final'],
    grid_params=None,
    seed=42,
        verbosity=10, knockout_tuner=None):
    """
    give default parameters and instantiate parameter generator
    """

    if stage == 'knockout':
        # default are four stages: vectorizer, transformer, selector, reducer
        grid_params = [
            {'reducer__estimator':
             [OneVsRestClassifier(LogisticRegression(random_state=seed)),
              OneVsRestClassifier(
                  LinearSVC(verbose=verbosity, random_state=seed))],
             'reducer__estimator_params':
             list(
                 ParameterGrid(
                     {'estimator__C': np.linspace(0.5, 1.5, 5),
                      'estimator__intercept_scaling':
                      np.linspace(0.5, 1.5, 5)}))},
            {'reducer__estimator': [OneVsRestClassifier(MultinomialNB())],
             'reducer__estimator_params':
             list(ParameterGrid(
                 {'estimator__alpha': np.logspace(-1, 1, 5)}))}]
        feature_params = {
            'selector__mode': [
                'fpr', 'fdr'], 'vectorizer__extractor__ngram_range': [
                (1, 1), (1, 2), (1, 3)]}
        knockout_params = {
            'scoring': [
                make_scorer(accuracy_score), make_scorer(
                    fbeta_score, beta=0.3), make_scorer(
                    precision_score, average='weighted')]}
        for param in grid_params:
            param.update(feature_params)
            param.update(knockout_params)
    elif stage == 'final':
        max_comps = 1000
        if max_comps > knockout_tuner.best_estimator_.named_steps[
                'reducer'].n_features_:
            max_comps = knockout_tuner.best_estimator_.named_steps[
                'reducer'].n_features_ - 1

        min_comps = max_comps // 10
        grid_params = [
            {'decomposer__n_components': np.linspace(min_comps, max_comps, 10),
             'classifier__estimator':
             [LDA(),
              SGDClassifier(n_iter=1, random_state=seed, n_jobs=-1,
                            verbose=verbosity)]}]
    return(ParameterGrid(grid_params))


def main(cv_num=5, seed=42, verbosity=10, data_range=None):
    """
    main training for feature seleciton
    """

    knockout_tuner = init_knockout(trainer.cv, seed, verbosity)

    for params in param_generator(stage='knockout'):
        knockout_result = split_train_parameters.call_and_shelve(
            knockout_tuner,
            params,
            X,
            y)
        knockout_tuner = knockout_result.get()

    with open('knockout_tuner.pickle', 'wb') as fp:
        joblib.dump(knockout_tuner, fp)

    final_tuner = init_final(knockout_tuner, trainer.cv, verbosity)
    for params in param_generator(stage='final'):
        final_result = split_train_parameters.call_and_shelve(
            final_tuner,
            params,
            knockout_tuner.best_estimator_.transform(X),
            y)
        final_tuner = final_result.get()

    with open('final_tuner.pickle', 'wb') as fp:
        joblib.dump(final_tuner, fp)

    return((knockout_tuner, final_tuner))


def construct_cv(ngrams, foldname='kfold', nfold=3, random_state=42,
              test_size=-1):
    """
    generate predefined cross-validation data based on the samples lengths (for
    the unequal length dataset)

    @param ngrams is a list which stores the training data have been split into
        n-grams phrases
    @param foldname is a string which is used to construct cross_validation
        instance 
    @param nfold is the fold number used to specify the number of fold used
    @param random_state is the random seed used in generating shuffling sample
    """

    # get a global index mapping for different lengths
    example_lens, lens_dist = recursnn_helper.compute_len_stats(ngrams)
    lens_dist = lens_dist[1:]

    if np.all(lens_dist) > 1:
        sampling_from_lens = np.hstack(
            list(
                map(lambda x, times: np.repeat(x, times), example_lens,
                    lens_dist)))
    # to deal with the cv exceptions which minimum number of sample cannot be
    # less than cv number or 2
    else:
        sampling_from_lens = np.hstack(
            list(
                map(lambda x, times: np.repeat(x, times),
                    example_lens[lens_dist > 1], lens_dist[lens_dist > 1])))

    fold_cv = None
    if foldname == 'kfold':
        fold_cv = StratifiedKFold(sampling_from_lens,
                                  n_folds=nfold, random_state=random_state)
    elif foldname == 'shufflesplit':
        if nfold > 1:
            raise NotImplementedError('cannot handle n_iter > 1 for now')
        tolerance = abs(test_size * (1 - test_size) - (np.sum(lens_dist >
            1) / np.sum(lens_dist))**2) > (1 / len(sampling_from_lens))

        test_size_ = (1 / nfold) * (test_size < 0) or \
            test_size * int(tolerance) or \
            int(np.sum(lens_dist > 1)) * int(not tolerance)
        assert(test_size <= 1.0 or type(test_size) == int)
        if test_size_ != test_size:
            logger.info("current test_size used is changed to "
                        "{:.2f} ({})".format(test_size_ / len(ngrams),
                                             int(test_size_)))
        fold_cv = StratifiedShuffleSplit(
            sampling_from_lens, n_iter=nfold, test_size=test_size_,
            random_state=random_state)
    else:
        raise NotImplementedError("cannot handle cross_validator type other"
                                  "other than kfold or shufflesplit")
    test_fold = -1 * np.ones((np.sum(lens_dist),))

    foldnum = 0
    for (fold_train, fold_test) in fold_cv:
        test_fold[fold_test] = foldnum
        foldnum += 1

    cv_iter = PredefinedSplit(test_fold=test_fold)
    return(cv_iter)

if __name__ == '__main__':
    main(cv_num=2, data_range=np.arange(1, 10))
