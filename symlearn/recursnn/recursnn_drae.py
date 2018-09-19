from sklearn.base import BaseEstimator
from sklearn.linear_model.base import LinearClassifierMixin
from sklearn.model_selection import check_cv 
from sklearn.utils import check_random_state

from symlearn.blocks.roles import WEIGHT, BIAS, AUXILIARY
from symlearn.blocks.filter import VariableFilter
from symlearn.blocks.algorithms import GradientDescent, Scale
from symlearn.blocks.main_loop import MainLoop
from symlearn.blocks.extensions import (FinishAfter, Printing, saveload, Timestamp,
                               Timing)
from symlearn.blocks.extensions.monitoring import (TrainingDataMonitoring, 
                                          DataStreamMonitoring)
from symlearn.blocks.bricks.cost import (BinaryCrossEntropy, CategoricalCrossEntropy,
                                MisclassificationRate, CostMatrix)
from symlearn.blocks.bricks import (NDimensionalSoftmax, MLP, Logistic, WithExtraDims)
from symlearn.blocks.bricks.base import application
from symlearn.blocks.model import Model
from symlearn.blocks.graph import ComputationGraph
from symlearn.blocks import initialization
from symlearn.fuel.streams import DataStream
from symlearn.fuel.schemes import SequentialScheme
from symlearn.fuel.datasets import IndexableDataset, IterableDataset
from symlearn.fuel.transformers import Mapping, Merge, Unpack
from theano import tensor
from string import Template

from functools import lru_cache, partial, wraps
from collections import namedtuple, OrderedDict
from itertools import zip_longest
from operator import itemgetter

from . import (recursnn_helper, CompilerABC, SharedAccess, timethis)
from .recursive import PenalizedLinear
from .recursnn_rae import RecursiveAutoEncoder
from .fuel_extensions import Stacking
from .. import  theano
from .. import blocks

import logging
import numpy as np
import scipy as sp
import types
import os

data_dir = os.path.join(os.getenv('WORKSPACE'), 'Kaggle', 'symlearn', 'data')

logger = logging.getLogger(__name__)

class ZeroOneLoss(CostMatrix):
    """
    modify blocks.bricks.cost.MisclassificationRate when the true lables, y, is
    two dimensional where the first dimension is the sub-labels for each
    training example, the second dimension is the ith example in one batch,
    rather one dimensional
    """
    def __init__(self, top_k, name=None):
        self.top_k = top_k
        super(ZeroOneLoss, self).__init__(name)
        
    @application(inputs=['y', 'y_hat'], outputs=['cost'])
    def cost_matrix(self, y, y_hat):
        """
        directly copy from symlearn.blocks.bricks.cost.MisclassificationRate without
        applying mean

        in order to be wrapped with wrapper brick, applicaiton must provide
        inputs property
        """
        # Support checkpoints that predate self.top_k
        top_k = getattr(self, 'top_k', 1)
        if top_k == 1:
            cost = tensor.neq(y, y_hat.argmax(axis=1))
        else:
            row_offsets = theano.tensor.arange(0, y_hat.flatten().shape[0],
                                               y_hat.shape[1])
            truth_score = y_hat.flatten()[row_offsets + y]
            # We use greater than _or equals_ here so that the model
            # _must_ have its guess in the top k, and cannot extend
            # its effective "list of predictions" by tying lots of things
            # for k-th place.
            higher_scoring = tensor.ge(y_hat, truth_score.dimshuffle(0, 'x'))
            # Because we used greater-than-or-equal we have to correct for
            # counting the true label.
            num_higher = higher_scoring.sum(axis=1) - 1
            cost = tensor.ge(num_higher, top_k)
        return(cost)


class NDimensionalErrorRate(ZeroOneLoss):
    decorators = [WithExtraDims()]


def check_labels(y):
    if isinstance(y, (list, tuple, np.ndarray)):
        return(y is not None and all([yy is None for yy in y]))
    else:
        raise ValueError('y must be an instance of list or numpy.ndarray')


class schedule_learning:
    def __init__(self, indices=None):
        self.indices = indices

    def __call__(self, X, y=None):
        """
        order the unequal length training examples by length 
        """
        Xt, indices = zip(*map(itemgetter(1, 2), sorted([(x.nnz, x, i)
            for i, x in enumerate(X)], key=itemgetter(0))))
        assert(not y is None)
        yt = [yy.reshape(-1, 1) if yy.ndim == 1 else yy for yy in y]
        yt = np.asarray([yt[idx] for idx in indices])
        if self.indices is None:
            self.indices = indices
        return(Xt, yt)


def _compute_loss(self, errors, handler):
    root_acc, nt_acc = [], []
    for err in errors:
        for e, acc in zip(err, [root_acc, nt_acc]):
            if handler is not None:
                e = handler(e)
            acc.append(e)
    if nt_acc:
        return(np.mean(root_acc), np.mean(nt_acc))
    else:
        return(np.mean(root_acc), np.nan)


def _score(self, X, y, scorerer='loss', **kwargs):
    scorerer = getattr(self, scorerer)
    if scorerer == self.error_rate: # using scorerer is self.error_rate doesn't
                                    # work
        def handle_error(val):
            if val > 1.0 and not np.isclose(val, 1.0):
                raise ValueError("Misclassification rate cannot exceed 1.0")
            return(1 - np.clip(val, 0.0, 1.0))
        handler = handle_error
    else:
        handler = None
    errors = scorerer(X, y, **kwargs)
    return(_compute_loss(self, errors, handler))


def proc_data(func):
    """
    module scoped unbound process data deliver training stream 
    """
    @wraps(func)
    def wrapper(self, X, y=None, **fit_params):
        dtype = fit_params.pop('dtype', 'int32')
        batch_size = fit_params.pop('batch_size', 1)
        raw_stream = Stacking(
            DataStream(IndexableDataset({'x': X}),
                iteration_scheme=SequentialScheme(
                    batch_size=batch_size, examples=len(X))),
            which_sources=('x',))

        # getting encoded features 
        x_primes = list(self.autoencoder.transform(raw_stream))
        encoding_stream = Stacking(DataStream(IndexableDataset({'x_prime': x_primes}),
                iteration_scheme=SequentialScheme(
                    batch_size=1, examples=len(x_primes))))
        if y is not None:         
            # transformed y label to have the same shape and batch size as
            # encoded features and turning into stream object
            y_primes = [(self.n_out // 2) *
                    np.ones(xx.shape[:-1]).astype(dtype) for i, xx in
                    enumerate(x_primes)]
            data_iter = zip(y, X)
            for i, this_ybatch in enumerate(y_primes):
                for j in range(batch_size):
                    yy, xx = next(data_iter)
                    if hasattr(yy, 'mask'):
                        yy = yy.data[~yy.mask].ravel()
                    else:
                        yy = yy[xx.diagonal() > 0].ravel()
                    this_ybatch[:len(yy), j] = yy 

            label_stream = Stacking(DataStream(IndexableDataset({'y': y_primes}),
                    iteration_scheme=SequentialScheme(
                        examples=len(y_primes), batch_size=1)))
            if not isinstance(func, theano.compile.function_module.Function):
                # func is fit method
                stream = Merge((label_stream, raw_stream), (('y', 'x')))
                return(func(self, stream, **fit_params))
            else:
                # func is loss computation method 
                stream = Merge((label_stream, encoding_stream), (('y', 'x_prime')))
                return(func(batch['x_prime'], batch['y']) for batch in
                    stream.get_epoch_iterator(as_dict=True))
        else: 
            # returning result should be a pair, the first is the probability
            # of node as the first dimension of the encoding inputs and the
            # second is the mask used for compute probability
            return(func(x['x_prime']) for x in
                    encoding_stream.get_epoch_iterator(as_dict=True))

    return(wrapper)


class RecursiveTreeClassifier(CompilerABC, LinearClassifierMixin, BaseEstimator):
    """
    factory method for create RecursiveTreeClassifier instance
    """
    allocate_args = ('optimizer', 'n_components', 'vocab_size', 'n_out')
    alpha = SharedAccess(0.0) 
    learning_rate = SharedAccess(0.1)

    def __init__(self, optimizer='GradientDescent', n_components=10,
            vocab_size=137, alpha=0.0, n_iter=1, learning_rate=0.01,
            init_weight=None, random_state=None, loss='log_loss',
            penalty='l2', fit_intercept=False, top_k=1, n_out=5):

       self.optimizer = GradientDescent
       self.n_components = n_components
       self.vocab_size = vocab_size
       self.alpha = alpha
       self.n_iter = n_iter
       self.learning_rate = learning_rate
       self.init_weight = init_weight 
       self.random_state = random_state
       self.loss = loss
       self.penalty = penalty 
       self.fit_intercept = fit_intercept
       self.top_k = top_k
       self.n_out = n_out

    def __new__(cls, *args, **kwargs): 
        """
        static method automatically without actually declare it
        """
        # pointing to object.__new__ and cls is mandatory argument should be
        # passed in (equivalently as object.__new__(cls))
        return(super(RecursiveTreeClassifier, cls).__new__(cls))

    def __str__(self):
        bricks = self.model.get_top_bricks()
        assert(isinstance(bricks[-1], MLP))
        return("%s(dim=[%d,%d])"%(self.__class__.__name__, *(bricks[-1].dims)))

    def set_params(self, **params):
        ae_params, sk_params = {}, {}
        model_params = self.model.get_parameter_values() 
        for name, val in params.items():
            snames = name.split('__', maxsplit=1)
            if snames[0] == 'autoencoder':  # set autoencoder
                ae_params[snames[1]] = val
            elif snames[0] == 'model':  # symbolic parameters
                assert(isinstance(val, dict))
                model_params.update(val)
            else:
                sk_params[name] = val
        self.autoencoder.set_params(**ae_params)
        self.model.set_parameter_values(model_params) 
        return(super(RecursiveTreeClassifier, self).set_params(**sk_params))

    def instantiate(self, *args, **kwargs):
        namespace = kwargs.copy()
        if args:
            optimizer, n_components, vocab_size, n_out = args
        else:
            for argname in self.allocate_args:
                namespace[argname] = kwargs.pop(argname)

        compiled_ns = self.compile(namespace['optimizer'],
                namespace['n_components'], namespace['vocab_size'],
                namespace['n_out'], **kwargs)
        assert(isinstance(compiled_ns, dict) and len(compiled_ns) > 0)

        # to make bound methods for compiled function 
        for k, v in compiled_ns.items():
            if isinstance(v, (theano.compile.function_module.Function,
                types.FunctionType)):
                namespace[k] = types.MethodType(v, self)
            else:
                namespace[k] = v

        namespace['score'] = types.MethodType(_score, self)

        # avoid manipulate instance.__dict__ but using setattr to guarantee
        # attributes are accessiable
        for k, v in namespace.items():
            setattr(self, k, v)
        return(tuple(namespace[argname] for argname in self.allocate_args))

    def compile(self, optimizer, n_components, vocab_size, n_out, **kwargs):
        # need to popout before passing into autoencoder
        top_k = kwargs.pop('top_k')
        loss = kwargs.pop('loss') 
        activations = kwargs.pop('activations', [Logistic(name='sigmoid')])
        indifferentiables = kwargs.pop('indifferentiables', None)

        autoencoder = RecursiveAutoEncoder(optimizer, n_components,
                vocab_size, **kwargs)

        # ensure using the same seed specified by user
        random_generator = check_random_state(kwargs.pop('random_state')) 
        init_weight = kwargs.pop('init_weight')
        if init_weight is None: 
            init_weight = initialization.IsotropicGaussian().generate(
                    random_generator, (n_components, n_out))
        assert(isinstance(init_weight, np.ndarray))
        init_weight = initialization.Constant(init_weight)

        # prepare the parameters used to construct MLP layers
        self.alpha = kwargs.pop('alpha') 
        penalty = kwargs.pop('penalty')
        fit_intercept = kwargs.pop('fit_intercept')

        # construct sigmoid / classifer layer
        brick = MLP(activations=activations,
                dims=[n_components, n_out],
                prototype=PenalizedLinear(penalty=penalty),
                weights_init=init_weight,
                use_bias=fit_intercept)

        # compute cost used for classifier layer 
        # two losses will be used: 
        # validate loss is used for cross-validate model and shouldn't be used
        # as the objective of optimization for parameter estimation 
        losses = OrderedDict() 
        losses['validate'] = NDimensionalErrorRate(
                top_k=top_k, name='validate__error').cost_matrix
        # object loss is used as the objective of optimization for parameter
        # estimation
        if loss == 'log_loss':
            losses['objective'] = NDimensionalSoftmax(
                    name='objective__%s' % loss).categorical_cross_entropy

        # prepare the parameters used to construct structural cost based on the
        # prior knowledge of parsing tree in order to minize the noise from
        # bottom layer passing through 
        dtype = kwargs.pop('dtype', 'int32')

        x_prime = autoencoder.model.outputs[0]
        #x_prime = tensor.matrix('x_prime', dtype=theano.config.floatX)
        x_prime.name = 'x_prime'
        y = tensor.matrix('y', dtype=dtype)
        y_hat = brick.apply(x_prime)

        assert(tuple(losses.keys()) == ('validate', 'objective'))
        assert(all(isinstance(app.brick, los) for app, los in
            zip(losses.values(),
                (ZeroOneLoss, blocks.bricks.Softmax))))

        # compute not-weighted and not-discounted raw costs 
        costs = [func(y, y_hat, extra_ndim=1) for _, func in losses.items()]
        costs[0].name = losses['validate'].brick.name
        costs[-1].name = losses['objective'].brick.name

        cost_spec = kwargs.pop('cost_spec', 'root_only')

        # only loss at the root of parsing tree will be used for optimization
        # (with weight as 1.0 and the rest is zeros) if cost_spec specified is
        # "root_only"
        cost_weight = tensor.zeros(x_prime.shape[:-1],
                dtype=theano.config.floatX)
        cost_weight = tensor.set_subtensor(cost_weight[0], 1.0)
        if cost_spec != 'root_only':
            # otherwise, include the lower layers cost with different weighting
            # scheme
            #node_weight = tensor.tensor4D(dtype=theano.config.floatX) 
            node_weight = VariableFilter(
                    roles=[blocks.roles.INPUT], 
                    bricks=autoencoder.model.get_top_bricks(),
                    name=cost_spec)(autoencoder.model.variables)
            assert(len(node_weight_) == 1)
            if cost_spec == 'span_range':
                # applying non-uniform weights based on the nodes included
                # below
                cost_weight = tensor.cast(node_weight[0][:, :, 1] - node_weight[0][:, :, 0],
                        dtype=theano.config.floatX)
                cost_weight /= tensor.shape_padleft(cost_weight[0])
                cost_weight = tensor.set_subtensor(cost_weight[1:],
                        cost_weight[1:] / cost_weight[1:].sum(axis=0,
                            keepdims=True))
            elif cost_spec == 'mask':
                # applying uniform weights only discounting those masked
                cost_weight = 1.0 - tensor.cast(node_weight[0], theano.config.floatX)
                cost_weight = tensor.set_subtensor(cost_weight[1:],
                        cost_weight[1:] / cost_weight[1:].sum(axis=0,
                            keepdims=True))
        
        details = OrderedDict.fromkeys(losses)
        for i, loss in enumerate(costs[:]):
            # applying cost_weights to raw costs
            elemwise = loss * cost_weight
            losstype, lossname  = loss.name.split('__')
            details[losstype] = [
                    ('root', elemwise[0].mean()),
                    ('not_root', elemwise[1:].sum(axis=0).mean())]
            costs[i] = tensor.stack(
                    [v for (k, v) in details[losstype]]).sum()
            costs[i].name = lossname

        # taking differentiation
        self.__model= Model(costs)

        parameters = VariableFilter(
                roles=[blocks.roles.PARAMETER])(self.__model.variables)
        
        # adding regularization
        aux = [v for p in parameters if blocks.roles.WEIGHT in p.tag.roles 
                 for v in p.tag.annotations[0].auxiliary_variables
                 if v.name.index('norm') > 0]

        if self.alpha > 0.:
            penalized_loss = self.__model.outputs[-1] + (self.alpha *
                    tensor.ones(len(aux), dtype=theano.config.floatX) *
                    tensor.stack(aux, axis=0)).sum()
            penalized_loss.name = 'penalized_objective'
            self.__model.outputs.append(penalized_loss)

        if not brick.initialized:
            brick.initialize()
        
        indifferentiables = VariableFilter(
                roles=[blocks.roles.INPUT],
                bricks=autoencoder.model.get_top_bricks())(autoencoder.model.variables)
        newkws = {'consider_constant': indifferentiables}
        optimizer_ = optimizer(cost=self.__model.outputs[-1],
                                    parameters=parameters,
                                    step_rule=Scale(
                                        learning_rate=self.learning_rate),
                                    **newkws)

        # prepare namespace for new instance
        compiled_ns = {}
        compiled_ns['predict_proba'] = proc_data(theano.function([x_prime],
                [y_hat, tensor.shape_padright(cost_weight)],
                name="%s.%s"%(self.__class__.__name__, '_predict_proba')))

        if cost_spec == 'root_only':
            compiled_ns['error_rate']= proc_data(theano.function([x_prime, y],
                    [self.__model.outputs[0]],
                    name="%s.%s"%(self.__class__.__name__, '_error_rate')))
            compiled_ns['loss'] = proc_data(theano.function([x_prime, y], 
                    [self.__model.outputs[-1]],
                    name="%s.%s"%(self.__class__.__name__, '_loss')))
        else:
            compiled_ns['error_rate'] = proc_data(theano.function([x_prime, y],
                    list(map(itemgetter(1), details['validate'])),
                    name="%s.%s"%(self.__class__.__name__, '_error_rate')))
            compiled_ns['loss'] = proc_data(theano.function([x_prime, y], 
                    list(map(itemgetter(1), details['objective'])),
                    name="%s.%s"%(self.__class__.__name__, '_loss')))
        compiled_ns['fit'] = proc_data(self.driver(optimizer_,
            variables=[self.__model.outputs[-1],
                autoencoder.model.outputs[-1]] + aux))
        #    ] + aux))
        compiled_ns['autoencoder'] = autoencoder
        compiled_ns['random_state_'] = random_generator
        compiled_ns['model'] = self.__model
        compiled_ns['optimizer_'] = optimizer_
        logger.info("%s compiling functions with brick initialized"
                %(self.__class__.__name__))
        return(compiled_ns)

    def driver(self, optimizer_, **kwargs): 
        """
        return a MainLoop instance prebound to optimizer built through compile
        method and only leaving data_stream for the client code
        """
        variables = kwargs.pop('variables', [self.__model.outputs[-1]])
        main_loop = partial(MainLoop, algorithm=optimizer_, model=self.__model)
         
        @wraps(main_loop)
        def loop_wrapper(self, training_stream, **fit_params):
            backend = fit_params.get('log_backend')
            save_models = fit_params.get('save_model', True)
            default_extensions = [
                    TrainingDataMonitoring(variables=variables,
                                           prefix=str(self),
                                           after_batch=True),
                    FinishAfter(after_n_epochs=self.n_iter),
                    Timestamp(), Printing()]
            if save_models: 
                default_extensions += [
                        saveload.Checkpoint(
                            os.path.join(data_dir, '%s.model' % (str(self))),
                    save_separately=None, use_cpickle=False),
                        saveload.Load(
                            os.path.join(data_dir, '%s.model' % (str(self))),
                    load_iteration_state=False, load_log=False)]
            logger.info('start supervised fine-tuning step ({} with batch_size '
                    '= {})'.format(str(self),
                        training_stream.data_streams[-1].data_stream.iteration_scheme.batch_size))
            main_loop(data_stream=training_stream,
                    extensions=default_extensions,
                    log_backend=backend).run()
            logger.info('complete supervised fine-tuning step %s'%
                    str(self))
            return(self)
        return(loop_wrapper)
