from sklearn.feature_extraction.text import CountVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
from nltk.tokenize.treebank import TreebankWordTokenizer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer

from functools import partial

from symlearn.utils import VocabularyDict

import numpy
import re
import scipy
import cython

NUM_PAT = re.compile(r"([()+\-.,]?\d+)+")
PUNKT_PAT = re.compile(r"^([^\w]+)$")
PROC_PAT = re.compile(r'\\/')


def process(raw, tokenizer=lambda x: x.split(), norm_number=None,
        norm_punkt=None, morphier=None):
    doc = PROC_PAT.sub(r'/',raw.lower())
    if norm_number and norm_punkt and not morphier:
        return(list(tokenizer(doc)))
        
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