    
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
from symlearn.utils import (VocabularyDict, count_vectorizer, construct_score, 
    inspect_and_bind)
from gensim.sklearn_api.w2vmodel import W2VTransformer
from joblib.numpy_pickle import NumpyUnpickler, NumpyPickler

import os
import joblib


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
    return preprocessor


def load_from_word2vec(pretrain_loc, vocab):
    """
    original word2vec is quite big and contain a lot of extra vocabulary which is not contained 
    within current training corpus 
    """
    filename = pretrain_loc.as_posix()
    with open(filename, 'r+b') as fp:
        mmap_fp = mmap.mmap(fp.fileno(), 0)
        raw_w2v = kv.KeyedVectors.load_word2vec_format(mmap_fp, 
                    binary=True, limit=10000)
    proc_w2v = kv.Word2VecKeyedVectors(raw_w2v.vector_size)
    proc_w2v.vectors = numpy.zeros((len(vocab), raw_w2v.vector_size), 
        dtype=raw_w2v.vectors.dtype)
    # construct word<->freq dict
    proc_w2v.build_vocab_from_freq()
    vectorizer= W2VTransformer().fit(X=None)
    vectorizer.gensim_model.wv = word2vec
    n_components = vectorizer.gensim_model.wv.vector_size
    # adding extra one column for oov
    vectorizer.gensim_model.wv.vectors = numpy.vstack(
        [vectorizer.gensim_model.wv.vectors, 
        numpy.zeros(n_components, dtype=vectorizer.gensim_model.wv.vectors.dtype)])
    # update word2vec.vocab
    new_vocab = dict([(v, word2vec.vocab[k]) if k in word2vec.vocab else 
        (v, kv.Vocab(index=vocab[k], count=None)) for k, v in vocab.items()])
    word2vec.vocab = new_vocab
    pretrain_loc = pretrain_loc.parent.joinpath('word2vec.bin')
    word2vec.save(pretrain_loc.as_posix())
    word2vec = kv.KeyedVectors.load(pretrain_loc.as_posix(), mmap='r')
    
    return vectorizer, track_index


class RestrictedUnpickler(NumpyUnpickler):
    """
    overwrite find_class to restrict global modules import
    """

    def find_class(self, module, name):
        """
        replace the outdated packages with more update ones

        Parameters:
        -----------
        see python official document
        """
        import pickle, sys
        # Only allow safe classes from builtins.
        if not (module in globals() or module in sys.modules):
            if str(module) == '_aux':
                module = sys.modules['aux']
            if name in ['count_vectorizer', 'WordNormalizer', 'VocabularyDict',
            'construct_score', 'inspect_and_bind']:
                logging.warn('skipping importing rquired module %s because not found' % module)
                module = 'symlearn.utils'
            else:
                raise pickle.UnpicklingError("global '%s.%s' is forbidden" %
                    (module, name))
        return super(RestrictedUnpickler, self).find_class(module, name)

    def persistent_load(self, persistency):
        """
        pid should be the model name
        """
        cls, pid = persistency
        try:
            model = spacy.load(pid)
        except:
            raise pickle.UnpicklingError("cannot unpickle from persistent id")
        if cls == spacy.vocab.Vocab:
            return model.vocab
        else:
            return model


def patch_pickled_preproc(pretrain_loc):
    if os.path.exists(os.path.join(data_dir, pretrain_loc)):
        pretrain_loc = pathlib.Path(os.path.join(data_dir, pretrain_loc))
        vocab = VocabularyDict(os.path.join(data_dir, 'treebased_phrases_vocab'))
        if pretrain_loc.suffix.endswith('model'):
            # trained by unigram + truncatedSVD
            from unittest.mock import patch
            with patch.multiple(joblib.numpy_pickle, NumpyPickler=LanguageModelPickler, 
                NumpyUnpickler=RestrictedUnpickler, spec_set=True):
                preproc = joblib.load(pretrain_loc.as_posix())
            vocab = preproc.named_steps['vectorizer'].func.vocab
            # from gensim
        elif pretrain_loc.suffix.endswith('bin'):
            vectorizer = load_from_word2vec(pretrain_loc, vocab)
            preproc = Pipeline([('transform', vectorizer)])
        elif pretrain_loc.startswith('en'):
            # loading model from Spacy
            vocab, preproc = load_from('spacy', pretrain_loc)
            n_components = preproc.named_steps['transformer'].func.max_features
    elif pretrain_loc == 'train':
        # training word embedding on the fly
        preproc = construct_preprocessor(vocabulary=vocab)
        n_components = preproc.named_steps['decomposer'].n_components
    return preproc

