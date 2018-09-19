from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted
from sklearn.decomposition import TruncatedSVD
from nltk.tree import Tree

from operator import itemgetter
from itertools import chain


from symlearn.recursnn import recursnn_utils


import symlearn.csgraph.adjmatrix as pyadj
import numpy
import scipy
import logging
#import shelve
import os

logger = logging.getLogger(__name__)

data_dir = os.path.join(os.getenv('WORKSPACE'), 'Kaggle','symlearn', 'data')

from nltk.tokenize.treebank import TreebankWordTokenizer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer

from functools import partial


from symlearn.utils import VocabularyDict 

import numpy
import spacy
import re
import scipy
import cython
import warnings


NUM_PAT = re.compile(r"([()+\-.,]?\d+)+")
PUNKT_PAT = re.compile(r"^([^\w]+)$")
PROC_PAT = re.compile(r'\\/')


def process(raw, tokenizer=lambda x: x.split(), norm_number=None,
        norm_punkt=None, morphier=None):
    doc = PROC_PAT.sub(r'/',raw.lower())
    if norm_number and norm_punkt and not morphier:
        return(list(tokenizer(doc)))
    words = []    
    for word in tokenizer(doc): # assume single doc 
        if norm_number:
            word = NUM_PAT.sub("-num-", word)
        if norm_punkt:
            word = PUNKT_PAT.sub("-punkt-", word)
        if morphier:
            word = self.morphier(word)
        if len(word) > 0:
            words.append(word)
    return(words)


class WordNormalizer(object):
    
    def __init__(self, tokenizer=None, norm_number=True, norm_punkt=True,
            norm_morphy=None):
        """
        word normalizer used after tokenizer and before vectorizer/transformer
        """
        morphiers = {'stem': PorterStemmer().stem,
                     'lemma': WordNetLemmatizer().lemmatize}
        
        if tokenizer == None:
            tokenizer = TreebankWordTokenizer().tokenize
        self.tokenizer = tokenizer
        
        self.norm_number = norm_number
        self.norm_punkt = norm_punkt
        self.morphier = None
        if norm_morphy:
            try:
                self.morphier = morphiers[norm_morphy]
            except:
                print("key error! using None")
                pass
    
    def __call__(self, raw_doc):
        """
        serve as tokenizer in CountVectorizer
        >>> [['a','dog'],['one','cat']] == WordNormalizer()(['a dog','one cat'])
        ... True
        >>> ['a','dog','and','one','cat'] == WordNormalizer()('a dog and one cat')
        ... True
        """
        if type(raw_doc) is str:
            raw_doc = [raw_doc]
        func = partial(process, tokenizer=self.tokenizer,
                norm_number=self.norm_number, norm_punkt=self.norm_punkt,
                morphier=self.morphier) 
        docs = list(map(func, raw_doc))
        if len(raw_doc) == 1:
            return(docs.pop())
        return(docs)
        

class WordNormalizerWrapper(BaseEstimator, TransformerMixin):
    def __init__(self, extractor=None, normalizer=None, norm_kwargs={},
            ext_kwargs={}):
        
        if not normalizer:
            self.normalizer = WordNormalizer()
        elif normalizer is type:
            self.normalizer = normalizer(**norm_kwargs)
        else:
            self.normalizer = normalizer
        
        if not extractor:
            self.extractor = CountVectorizer(
                    tokenizer=self.normalizer.transform, strip_accents='ascii',
                    max_df=0.5)
        elif extractor is type:
            self.extractor = extractor(tokenizer=self.normalizer.transform,
                    **ext_kwargs)
        else:
            self.extractor = extractor
            
    def fit(self, raw_doc, y = None, **fit_params):
        return(self.extractor.fit(raw_doc))
    
    def partial_fit(self, raw_doc, y = None, **fit_params):
        return(self.extractor.partial_fit(raw_doc))

    def transform(self, raw_doc, copy = True):
        return (self.extractor.transform(raw_doc))


@cython.cclass
class count_vectorizer(object):
    """
    a simplified scikit-learn CountVectorizer or stateful scikit-learn
    DictVectorizer which owns a VocabularyDict instance and other possible
    keywords to customize VocabularyDict
    """
    cython.declare(vocab=VocabularyDict, dtype=numpy.dtype)
    def __init__(self, vocab_file, max_features=numpy.Inf, dtype=numpy.float32):
        """
        Parameters
        ----------
        @param vocab_file: VocabularyDict instance or str
            the filename used to open VocabularyDict instance (using
            shelve.open) or a reference of an external VocabularyDict instance
        @param kwargs: dict and possible keywords are:
            max_features: int
                used to indicate the maximal vocabulary size for
                re-constructing a trucated vocabuarly dict. Any vocaublary
                whose index (zero-based) is greater or equal to max_features
                will be re-assigned to **-unk-** 
            dtype: the type used to construct array based on vocabulary count
        """
        self.dtype = dtype
        if type(vocab_file) is str:
            vocab = VocabularyDict(vocab_file, max_features=max_features)
        else:
            vocab = vocab_file
        # assert(numpy.max(list(vocab.values())) == len(vocab))
        self.vocab = vocab

    # cython: annotation_typing=True
    def __call__(self, X:list, y=None)->scipy.sparse.csr_matrix:
        data, rows, cols = [], [], [] 
        for i, words in enumerate(X):
            if type(words[0]) is str:
                words = [self.vocab[w] for w in words]
            idx, cnts = numpy.unique(words, return_counts=True)
            idx[idx==-1] = len(self.vocab) - 1
            data.append(cnts)
            cols.append(idx)
            rows.append(numpy.repeat(i, len(idx)))

        return scipy.sparse.coo_matrix((numpy.hstack(data),
            (numpy.hstack(rows), numpy.hstack(cols))), shape=(len(X),
                len(self.vocab)), dtype=self.dtype).tocsr() 


class spacy_vectorizer(object):
    """
    a proxy class of spacy.Language model instance acting like a scikit-learn 
    Pipeline object handling several consecutive transformations and return a
    average word embedding vector for each sentence
    """

    def __init__(self, model_file, max_features=numpy.Inf, dtype=numpy.float32):
        """
        load the spacy model from the specified model_file

        Parameters
        ----------
        @param model_file: string 
            the model name or path for spacy to load
        @param max_features: int
            to scale the dimensionality of vector 
        @param dtype: numpy.dtype object
            to specify the data type for the word embedding
        """
        self.model = spacy.load(model_file)
        self.dtype = dtype

        if max_features < self.model.vocab.vectors_length:
            self.model.vocab.resize_vectors(max_features)

        self.max_features = self.model.vocab.vectors_length
            
    def preprocess(self, rawdocs:list):
        """
        process text and return doc object for storing

        Parameters
        ----------
        """
        for i, text in enumerate(rawdocs):
            if isinstance(text, list):
                text = ' '.join(text)
            text = text.replace('\\', '') # dealing with backslash
            yield self.model.make_doc(text)

    def __call__(self, X:list, y=None)->numpy.array:
        """
        convert a list of text into vectors 

        Parameters
        ----------
        @param X: list of text
            the collection of phrases in natural language for processing
        @param y: list of int
            the target labels to be passed along if needed; usually has no effect
        """
        # allocate array for the embeddings
        Xt = numpy.zeros((len(X), self.max_features), dtype=self.dtype)
        if isinstance(X[0], (str, list)):
            iter_ = self.preprocess(X)
        else:
            iter_ = enumerate(X)

        with warnings.catch_warnings():
            warnings.simplefilter("error", category=RuntimeWarning)
            for i, doc in iter_:   
                if doc.vector_norm != 0: # not in the vocabulary, will be all zeros 
                    Xt[i] = doc.vector / doc.vector_norm
        return Xt


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
