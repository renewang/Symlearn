from sklearn.pipeline import Pipeline
from sklearn.preprocessing import label_binarize
from sklearn.utils.validation import check_is_fitted
from sklearn.feature_extraction.dict_vectorizer import DictVectorizer
from sklearn.exceptions import NotFittedError
from sklearn.metrics import brier_score_loss

from copy import deepcopy
from itertools import groupby
from symlearn.utils import VocabularyDict
from operator import itemgetter

from stanfordSentimentTreebank import preprocess_data

import typing
import logging
import pandas

import scipy
import joblib
import gc
import os

import cython

import numpy
cimport numpy


cpdef typing.Iterable process_joint_features(tuple data, DictVectorizer vectorizer=*, 
    int n_levels=*, int n_classes=*)
  
cpdef pandas.DataFrame transform_features(str csv_file, int n_rows=*, Pipeline preproc=*, 
    VocabularyDict vocab=*)

cpdef list group_fit(numpy.ndarray levels, numpy.ndarray phrases, numpy.ndarray sentiments, Pipeline preproc, 
    list estimators, int max_level)

cpdef class labels_to_attributes(object):

    cpdef typing.Iterable __call__(self, pandas.DataFrame raw_data, list y=*, 
        list label_predictors=*)