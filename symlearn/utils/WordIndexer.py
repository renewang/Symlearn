from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from nltk.tree import Tree

from operator import itemgetter
from itertools import chain


from symlearn.recursnn import recursnn_utils
from symlearn.utils import (WordNormalizer, VocabularyDict)

import symlearn.csgraph.adjmatrix as pyadj
import numpy
import scipy
import logging
#import shelve
import os

logger = logging.getLogger(__name__)

data_dir = os.path.join(os.getenv('WORKSPACE'), 'Kaggle','symlearn', 'data')

class EstimatorAdaptor(BaseEstimator):

    def __init__(self, adaptee, **kwargs):
        assert(hasattr(adaptee, 'partial_fit'))
        self.adaptee = adaptee
        super(__class__, self).__init__(**kwargs)

    def fit(self, X, y, **fit_params):
        n_samples = self._get_n_samples(X)
        for i in range(n_samples):
            self.adaptee.partial_fit(X[i], y[i], **fit_params)
        return(self)

    def predict(self, X):
        result = []
        n_samples = self._get_n_samples(X)
        for i in range(n_samples):
            result.append(self.adaptee.predict(X[i]))
        return(numpy.hstack(result))

    def score(self, X, y):
        yt = numpy.asarray(list(chain.from_iterable(y)))
        predicted = self.predict(X)
        return(numpy.mean(predicted == yt))

    def _get_n_samples(self, X):
        return(itemgetter(0)(getattr(X, 'shape', [None])) or len(X))


class WordIndexer(object):
    """
    building greedy trees if input are documents and then construct adjacent
    matrices to represent connectivity in trees or given tree structures (in
    penn tree format) return adjacent matrices
    """
    def __init__(self, preprocessor):
        """
        Parameters
        ----------
        @param preprocessor: scikit-learn.feature_extraction TransformerMixin
                             instance
            word vectorizer to handle word preprocessings
        """
        is_norm_num = not (
                preprocessor.named_steps['vectorizer'].func.vocab.norm_number
                is None)
        is_norm_punkt = not (
                preprocessor.named_steps['vectorizer'].func.vocab.norm_punkt
                is None)
        self.preprocessor = preprocessor
        self.analyzer = WordNormalizer(
                tokenizer=lambda x: x.strip().lower().split(),
                norm_number=is_norm_num, norm_punkt=is_norm_punkt)

    def _is_convertible(self, input_str):
        """
        Parameters
        ----------
        check if the input_str is convertible to nltk tree instance

        @param input_str: string
            either in tree format for parsing or pure string
        """
        if type(input_str) == bytes:
            input_str = input_str.decode(encoding='utf-8')
        try:
            inst = Tree.fromstring(input_str)
        except ValueError as e:
            logger.info("{}".format(e))
            return None
        else:
            return inst

    def _proc_mats(self, X, y, **fit_params):
        vocab = self.preprocessor.named_steps['vectorizer'].func.vocab
        index2word = numpy.asarray(list(map(itemgetter(1), sorted([(v, k) for k, v
                                in vocab.items()], key=itemgetter(0)))))
        adjmats = [pyadj.to_csgraph(s, vocab, preprocessor=self.analyzer)
            for s in X]
        return(adjmats, [mat.delegators['descriptor']['labels'] for mat in
            adjmats])

    def __call__(self, X, y=None, **fit_params):
        """
        conduct regular fit process if X are pure string; otherwise, defer fit
        process in transform

        Parameters
        ----------
        @param X is a homogeneous matrix which is either in pure string or
                    string in tree format
        @param y is a matrix for labels in classification problem
        @param fit_params for other fit parameters passing along
        """

        # determined the incoming training set is in tree format or pure string
        proc_trees = None
        assert(self._is_convertible(X[0]))  # are tree-convertible string
        Xt, yt = self._proc_mats(X, y, **fit_params)

        if y is not None and len(y) > 0 and y[0] is None:
            assert(len(yt) >= len(y))
            for i in range(len(y)):
                y[i] = yt[i]
            if len(yt) > len(y):
                y.extend(yt[len(y):])

        return([x.tocsr() for x in  Xt])
