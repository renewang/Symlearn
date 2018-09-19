import warnings
with warnings.catch_warnings():
    warnings.simplefilter('ignore', DeprecationWarning)
    from sklearn.base import BaseEstimator
    from sklearn.base import TransformerMixin
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.pipeline import FeatureUnion
    from sklearn.externals.joblib import Parallel, delayed
    from sklearn.pipeline import (_fit_one_transformer, _fit_transform_one, 
                                  _transform_one)
    import sklearn

from itertools import chain
import numpy as np
import scipy as sp
import pandas as pd


class SoftMaxScaler(BaseEstimator, TransformerMixin):

    def __init__(self, copy=True,  scale_func='hyperbolic', pre_centered=False):
        ''' constructor '''

        self.copy = copy
        self.scaler = StandardScaler(copy=copy)
        # func_choices = {'sigmoid': expit, 'hyperbolic': np.tanh} <- not
        # working
        self.scale_func = scale_func
        self.pre_centered  = pre_centered

    def fit(self, X, y=None):
        '''fit'''
        if not self.pre_centered:
            self.scaler = self.scaler.fit(X)
        return(self)

    def transform(self, X, y=None, copy=None):
        '''transformation'''
        if sp.sparse.issparse(X):
            Xt = X.toarray()
        else:
            Xt = X
        if not self.pre_centered:
            Xt = self.scaler.transform(Xt, copy=copy)
        if self.scale_func == 'hyperbolic':
            return(np.tanh(Xt/2))
        elif self.scale_func == 'sigmoid':
            return(sp.special.expit(Xt))

    def inverse_transform(self, X, copy=None):
        '''TODO'''

        pass


class IdentityTransformer(BaseEstimator, TransformerMixin):
    '''pass along the result'''

    def __init__(self):
        pass

    def fit(self, X, y=None):
        '''fit'''

        return(self)

    def transform(self, X, y=None, copy=None):
        '''transform'''
        if sp.sparse.issparse(X):
            return(X.toarray())
        return(X)


class PartialFitPipeline(Pipeline):

    def __init__(self, steps):
        super(PartialFitPipeline, self).__init__(steps)

    def _pre_transform(self, X, y=None, **fit_params):
        pack = False
        if isinstance(X[0], (list, np.ndarray)):
            Xt = list(chain.from_iterable(X))
            pack = True

        Xt, fit_params = super(PartialFitPipeline, self)._pre_transform(
                Xt, y, **fit_params)
        if pack:
            Xt = X
            for name, transformer in self.steps[:-1]:
                Xt = [transformer.transform(x) for x in Xt]
        return(Xt, fit_params)

    def transform(self, X):
        if isinstance(X[0], (list, np.ndarray)):
            Xt = X
            for name, transformer in self.steps[:-1]:
                Xt = [transformer.transform(x) for x in Xt]
            return(Xt)
        else:
            return(super(PartialFitPipeline, self).transform(X))

    def partial_fit(self, X, y=None, **fit_params):

        # calling the pre-transform to transform this batch data by trained
        # transformers its actually a dummy partial fit. Most of work will be
        # handled by HashVectorizer which will get and set data in the batch
        # size. TfIdf and TruncatedSVD will just treat the batch as the full
        # data
        # TODO: more should be done to achieve real on-line for Latent
        # Semantics Analysis. Ideally, a wrapper of gensim will be absolutely
        # needed.

        Xt, fit_params = self._pre_transform(X, y, **fit_params)
        # calling the partial_fit on transfromers in pipeline firstly
        for name, transform in self.steps:
            if hasattr(transform, 'partial_fit'):
                transform.partial_fit(Xt, y, **fit_params)
        return self

    def score(self, X, y=None):
        if isinstance(X[0], (list, np.ndarray)):
            Xt = self.transform(X)
            return(self.steps[-1][-1].score(Xt, y))
        else:
            return(super(PartialFitPipeline, self).score(X))


class ItemSelector(BaseEstimator, TransformerMixin):
    """For data grouped by feature, select subset of data at a provided key.

    The data is expected to be stored in a 2D data structure, where the first
    index is over features and the second is over samples.  i.e.

    >> len(data[key]) == n_samples

    Please note that this is the opposite convention to sklearn feature
    matrixes (where the first index corresponds to sample).

    ItemSelector only requires that the collection implement getitem
    (data[key]).  Examples include: a dict of lists, 2D numpy array, Pandas
    DataFrame, numpy record array, etc.

    >> data = {'a': [1, 5, 2, 5, 2, 8],
               'b': [9, 4, 1, 4, 1, 3]}
    >> ds = ItemSelector(key='a')
    >> data['a'] == ds.transform(data)

    ItemSelector is not designed to handle data grouped by sample.  (e.g. a
    list of dicts).  If your data is structured this way, consider a
    transformer along the lines of `sklearn.feature_extraction.DictVectorizer`.

    Parameters
    ----------
    key : hashable, required
        The key corresponding to the desired value in a mappable.
    """
    def __init__(self, key):
        self.key = key

    def fit(self, x, y=None):
        return self

    def transform(self, data):
        if type(self.key) == list:
            # output should be handed by DictVectorizer 
            if not isinstance(data, pd.DataFrame):
                data = pd.DataFrame(data)
            return data[self.key].to_dict(orient='records')
        # can be pipelined with different vectorizer 
        return np.asarray(data[self.key])


class FeatureCombiner(FeatureUnion):
  def __init__(self, transformer_list, combiner=None, n_jobs=1, transformer_weights=None):
    super(FeatureCombiner, self).__init__(transformer_list, n_jobs, transformer_weights)
    assert(combiner!=None)
    self.combiner = combiner
    
  def fit(self, X, y=None):
    super(FeatureCombiner, self).fit(X, y)
    return(self)
  
  def fit_transform(self, X, y=None, **fit_params):
    # cannot in a loop for multiprocessing backend
    with Parallel(n_jobs=self.n_jobs) as parallel:
        result = parallel(delayed(_fit_transform_one)(trans, name, X, y,
            self.transformer_weights, **fit_params)
          for name, trans in self.transformer_list)
    Xs, transformers = zip(*result)
    self._update_transformer_list(transformers)
    if any(sp.sparse.issparse(f) for f in Xs):
      Xs = sp.sparse.csr_matrix(self.combiner(*Xs))
    else:
      Xs = self.combiner(*Xs)
    return Xs
 
  def transform(self, X):
    with Parallel(n_jobs=self.n_jobs) as parallel:
        Xs = parallel(delayed(_transform_one)(trans, name, X,
            self.transformer_weights)
            for name, trans in self.transformer_list)
    if any(sp.sparse.issparse(f) for f in Xs):
        Xs = sp.sparse.csr_matrix(self.combiner(*Xs))
    else:
        Xs = self.combiner(*Xs)
    return Xs
  
  def get_params(self, deep=True):
    inherited_params = super(FeatureCombiner, self).get_params(deep=deep)
    params = {'combiner': self.combiner}
    params.update(inherited_params)
    return(params)
