from .base import (preprocess_dictionary, check_treebased_phrases, 
                   run_batch, tile_raster_images, compute_inverse,
                   VocabularyDict)
from .WordNormalizer import WordNormalizer, count_vectorizer

import inspect

def construct_score(y_true, y_pred, sample_weight=None):
    scores = numpy.zeros((2,))
    if not hasattr(y_true, 'mask'):
        scores[0] = accuracy_score(y_true, y_pred, normalize=True,
                sample_weight=sample_weight)
    else:
        if sample_weight:
            scores[0] = accuracy_score(y_true.data[y_true.mask], y_pred[y_true.mask],
                    normalize=True, sample_weight=sample_weight[y_true.mask])
            scores[1] = accuracy_score(y_true.data[~y_true.mask], y_pred[~y_true.mask],
                    normalize=True, sample_weight=sample_weight[~y_true.mask])
        else:
            scores[0] = accuracy_score(y_true.data[y_true.mask], y_pred[y_true.mask],
                    normalize=True, sample_weight=sample_weight)
            scores[1] = accuracy_score(y_true.data[~y_true.mask], y_pred[~y_true.mask],
                    normalize=True, sample_weight=sample_weight)
    return(scores)

def inspect_and_bind(unbound_func, **kwargs):
    """
    small helper function to bind unbound_func with arguments specified in
    kwargs

    Note: not hanlding VAR_POSITION and VAR_KEYWORD
    """
    func_argspec = inspect.signature(unbound_func)
    new_args = tuple(kwargs[name] for name, param in func_argspec.parameters.items()
            if (name in kwargs) and 
            (param.kind == inspect.Parameter.POSITIONAL_ONLY))
    new_kwargs = dict([(name, kwargs[name]) for name, param in func_argspec.parameters.items()
            if (name in kwargs) and
            (param.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD or
             param.kind == inspect.Parameter.KEYWORD_ONLY)])
    bound_args = func_argspec.bind(*new_args, **new_kwargs)
    return(bound_args)