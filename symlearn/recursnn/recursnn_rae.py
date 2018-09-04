from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_random_state, check_array

from itertools import chain
from abc import abstractmethod
from collections import deque, OrderedDict

from symlearn.blocks.bricks import (Sequence, FeedforwardSequence, Initializable,
                           application, Activation, Bias, Logistic)
from symlearn.blocks.algorithms import Scale, GradientDescent
from symlearn.blocks.bricks.cost import SquaredError
from symlearn.blocks import select, initialization
from symlearn.blocks.bricks.base import lazy
from symlearn.blocks.main_loop import MainLoop
from symlearn.blocks.extensions import (FinishAfter, Printing, saveload, Timestamp,
                               Timing)
from symlearn.blocks.extensions.monitoring import TrainingDataMonitoring
from symlearn.blocks.model import Model
from symlearn.blocks.filter import VariableFilter
from symlearn.blocks.roles import add_role
from symlearn.blocks.monitoring.evaluators import DatasetEvaluator 
from symlearn.fuel.streams import DataStream
from symlearn.fuel.datasets import IndexableDataset
from symlearn.fuel.schemes import SequentialExampleScheme, SequentialScheme
from symlearn.fuel.transformers import Transformer
from theano import tensor

from .fuel_extensions import Stacking, SourcewiseMapping
from .recursive import (TreeBrickWrapper, RecursiveBrick, LookupTree,
        PenalizedLinear)
from . import (split_params, inspect_and_bind, CompilerABC, SharedAccess,
                      timethis)
from functools import singledispatch, partial, wraps, update_wrapper
from copy import copy

import time
import types
import logging
import theano
import theano.sparse
import numpy
import scipy
import blocks
import weakref
import inspect
import os

data_dir = os.path.join(os.getenv('WORKSPACE'), 'Kaggle', 'symlearn', 'data')

logger = logging.getLogger(__name__)

class AbstractAutoEncoder(Sequence):
    """
    interface for AutoEncoder brick
    """
    def __init__(self, **params):
        """
        construct encoder and decoder bricks based on the passing parameters
        """
        init_params, applications = self._process_params(**params)
        self.enc_proxy = Encoder(**init_params['encoder'])
        self.dec_proxy = Decoder(**init_params['decoder'])
        applications = [
                self.enc_proxy.apply, self.dec_proxy.apply] + applications
        super(__class__, self).__init__(applications)

    @abstractmethod
    def _process_params(self, **params):
        """
        handling params for children bicks either split passed from the client
        code or create default when the params are not supplied through client
        also customize other children bricks
        """
        pass

    @abstractmethod
    def _encode(self, *args, **kwargs):
        """
        implement single step encode logics
        """
        pass

    @abstractmethod
    def _decode(self, *args, **kwargs):
        """
        implement single step encode logics
        """
        pass

    @abstractmethod
    def _reconstruct_cost(self, *args, **kwargs):
        """
        implement single step reconstruct cost
        """
        pass


class Transpose(Activation):
    """
    An activation brick which cannot allocate any shared memory but still can
    access weight, input_dim, output_dim as initializable brick through its
    transpose counterpart). Mostly used in tied weight scheme for autoencoder
    """

    @lazy(initialization=['ptr_param'])
    def __init__(self, ptr_param, **kwargs):
        self.ptr_param = ptr_param
        super(__class__, self).__init__(name=kwargs.get('name'))

    def _initialize(self):
        if not isinstance(
                self.ptr_param, tensor.sharedvar.TensorSharedVariable):
            raise ValueError('must provide an allocated TensorSharedVariable')
        self.parameters = [self.ptr_param.T]

    @property
    def input_dim(self):
        return self.ptr_param.shape[1]

    @property
    def output_dim(self):
        return self.ptr_param.shape[0]

    @application(inputs=['x'])
    def apply(self, x):
        return(tensor.dot(x, self.W))

    @property
    def W(self):
        return self.parameters[0]


class AutoEncoder(AbstractAutoEncoder):
    """
    single step auto-encoder for text auto-encoding
    """

    def __init__(self, dims, activations, **kwargs):
        self.dims = dims
        # setting encoder activations
        params = kwargs.copy()
        params['encoder__activations'] = activations
        params['encoder__input_dim'] = 2 * dims
        params['encoder__output_dim'] = dims
        super(__class__, self).__init__(**params)

    def _process_params(self, **params):
        """
        arrange bricks based on the parameters
        """
        init_params = split_params(**params)
        # setting decoder activations
        init_params['decoder']['input_dim'] = \
            init_params['encoder']['output_dim']
        init_params['decoder']['output_dim'] = \
            init_params['encoder']['input_dim']
        if init_params['decoder'].get('use_bias'):
            extra_bricks = [Bias(init_params['decoder']['output_dim'])]
        else:
            extra_bricks = []
        init_params['decoder']['activations'] = \
            init_params['encoder']['activations'] + extra_bricks
        applications = []
        applications.extend(chain.from_iterable(
            val.values() if key not in ['encoder', 'decoder'] else []
            for key, val in init_params.items()))
        return(init_params, applications)

    def _reconstruct_cost(self, parent, enc_mat, dec_mat, span_weight,
                          mapping=None, mask=None):
        """
        must direct decoded all from bottom to top. encoding result will pass
        to decoder immediately return reconstruction loss (will reset the
        dec_mat every call)
        """
        if mapping is None:
            raise(ValueError(
            "mapping must be passed in for current implementation"))

        batch_idx = tensor.arange(enc_mat.shape[1])
        children = mapping[parent, batch_idx]
        # sum along the feature axis because this is per node base iteration
        cost = tensor.stack([
            (span_weight[parent, batch_idx, 0] *
            tensor.sqr(
                enc_mat[children.take(0, axis=children.ndim - 1), batch_idx] - 
                dec_mat[children.take(0, axis=children.ndim - 1), batch_idx]).sum(axis=-1)),
            (span_weight[parent , batch_idx, 1] *
            tensor.sqr(
                enc_mat[children.take(1, axis=children.ndim - 1), batch_idx] -
                dec_mat[children.take(1, axis=children.ndim - 1), batch_idx]).sum(axis=-1))
            ]).sum(axis=0)
        cost.name = 'direct_cost'
        return(cost)

    def _encode(self, parent, word_mat, mapping=None):
        """
        @param parent is a tensor.scalar which will be used to indicate which
                        internal node to be applied encoding
        @param word_mat is a tensor.matrix and should be in tree_size x n_batch
                        x embedding space (or the length of word vector)
        @param mapping is a tensor matrix which will be used to indicate the
                        children nodes of non-terminal node
        @param mask is a tensor matrix which will masked the padded tree size
        """
        if mapping is None:
            raise(ValueError(
            "mapping must be passed in for current implementation"))

        if parent.ndim == 0:  # prevent 0d scalar case
            parent = parent.dimshuffle('x')

        if word_mat.ndim == 2:
            word_mat = tensor.shape_padaxis(word_mat, axis=1)

        if mapping.ndim == 2:
            mapping = tensor.shape_padaxis(mapping, axis=1)

        # n_batch x 2
        batch_idx = tensor.arange(word_mat.shape[1])
        children = mapping[parent, batch_idx] 

        # n_batch x 2*n_hiddens
        children_vectors = tensor.concatenate([
            word_mat[children.take(0, axis=children.ndim - 1), batch_idx],  # left child
            word_mat[children.take(1, axis=children.ndim - 1), batch_idx]], # right child
            axis=-1)
        return(tensor.inc_subtensor(word_mat[parent, tensor.arange(word_mat.shape[1])],
                self.enc_proxy.apply(children_vectors)))

    def _decode(self, parent, dec_mat, mapping=None, enc_mat=None):
        """
        @param parent is a tensor.scalar which will be used to indicate which
                        internal node to be applied encoding
        @param dec_mat is a tensor.matrix and should be in tree_size x n_batch
                        x embedding space (or the length of word vector)
        @param enc_mat is a tensor.matrix and should be in tree_size x n_batch
                        x embedding space (or the length of word vector)
        @param mapping is a tensor matrix which will be used to indicate the
                        children nodes of nonterm node
        @param mask is a tensor matrix which will masked the padded tree size
        """
        if mapping is None:
            raise(ValueError(
            "mapping must be passed in for current implementation"))

        if parent.ndim == 0:  # prevent 0d scalar case
            parent = parent.dimshuffle('x')

        if dec_mat.ndim == 2:
            dec_mat = tensor.shape_padaxis(dec_mat, axis=1)

        from_where = dec_mat

        if enc_mat:
            if enc_mat.ndim == 2:
                enc_mat = tensor.shape_padaxis(enc_mat, axis=1)
            from_where = enc_mat

        if mapping.ndim == 2:
            mapping = tensor.shape_padaxis(mapping, axis=1)

        # n_batch x 2
        batch_idx = tensor.arange(from_where.shape[1]) 
        children = mapping[parent, batch_idx]

        parental_vector = from_where[parent, batch_idx]

        # n_batch x 2
        decoded_vectors = self.dec_proxy.apply(parental_vector)
        dec_mat = tensor.inc_subtensor(
                dec_mat[children.take(0, axis=children.ndim - 1), batch_idx],
                        decoded_vectors[:, 0: self.dims])

        dec_mat = tensor.inc_subtensor(
                dec_mat[children.take(1, axis=children.ndim - 1), batch_idx],
                        decoded_vectors[:, self.dims: 2 * self.dims])
        return(dec_mat)

    def _unfold_cost(self, enc_mat, dec_mat, span_weight, mapping, mask):
        """
        unfolding cost cacluation through the root of tree and propagate the
        error to the node

        @param enc_mat is a tensor.matrix and should be in tree_size x n_batch
                        x embedding space (or the length of word vector)
        @param dec_mat is a tensor.matrix and should be in tree_size x n_batch
                        x embedding space (or the length of word vector)
        @param span_range is a tensor.matrix and will be used to assigned
                        weight for its own children node when computing
                        decoding loss
        @param mapping is a tensor matrix which will be used to indicate the
                        children nodes of nonterm node
        @param mask is a tensor matrix which will masked the padded tree size
        """

        batch_idx = tensor.arange(enc_mat.shape[1])
        cost_per_node = self.children[-1].cost_matrix(enc_mat, dec_mat).sum(
                             axis=-1, keepdims=True)
        filtered = tensor.shape_padright(tensor.cast(
                (1 - tensor.eq(mapping[:, :, 1] - mapping[:, :, 0], 0)),
                dtype=span_weight.type.dtype)) # filter terms
        ufcost = (filtered * span_weight * tensor.concatenate([
                cost_per_node[mapping[:, :, 0], batch_idx],  # weight left child
                cost_per_node[mapping[:, :, 1], batch_idx]], # weight right child
                axis=-1)).sum(axis=(0, -1)).mean()
        ufcost.name = 'unfold_cost'
        return(ufcost)

    @application(inputs=['parent', 'word_mat', 'mapping'],
                 outputs=['enc_mat'])
    def encode_apply(self, parent, word_mat, mapping=None):
        enc_mat = self._encode(parent, word_mat, mapping)
        return(enc_mat)

    @application(inputs=['parent', 'dec_mat', 'mapping', 'enc_mat'],
                 outputs=['dec_mat'])
    def decode_apply(self, parent, dec_mat, mapping=None, enc_mat=None):
        enc_mat = self._decode(parent, dec_mat, mapping, enc_mat)
        return(enc_mat)

    @application
    def apply(self, application_call, *args, **kwargs):
        """
        apply single step re-construction without propagating errors if unfold
        keywards is not given; otherwise, unfolding or propagating the
        reconstruction errors

        @param application_call: reserve for blocks to pass application
        decorator wrapper in order to store variable for later use
        """
        is_unfold = kwargs.get('unfold', True)
        if is_unfold:
            result = self._unfold_cost(*args)
        else:
            new_kwargs = kwargs.copy()
            encode_argspec = inspect_and_bind(self._encode, **new_kwargs)
            enc_mat = self._encode(*encode_argspec.args,
                                   **encode_argspec.kwargs)
            dec_mat = tensor.zeros_like(enc_mat)
            new_kwargs['enc_mat'] = enc_mat
            new_kwargs['dec_mat'] = dec_mat
            decode_argspec = inspect_and_bind(self._decode, **new_kwargs)
            dec_mat = self._decode(*decode_argspec.args,
                                   **decode_argspec.kwargs)
            new_kwargs.update({'dec_mat': dec_mat})
            cost_argspec = inspect_and_bind(self._reconstruct_cost,
                                            **new_kwargs)
            cost = self._reconstruct_cost(*cost_argspec.args,
                                          **cost_argspec.kwargs)
            result = [cost, enc_mat, dec_mat]
        return(result)


class Encoder(FeedforwardSequence, Initializable):
    """
    encoder brick
    """
    def __init__(self, activations=None, **kwargs):
        """
        arrange and group basic bricks to construct encoder

        @params activations is a list containing blocks.bricks with apply
        methods. Notice, for each activation brick containing in the
        activations list if not paried with an initializable brick will
        automatically insert Linear-like brick to pair with
        """
        # checking if there's Linear brick before Activation
        application_methods = []
        for i, act in enumerate(activations):
            if hasattr(act, 'apply'):
                # only include objects with apply method in activations list 
                if isinstance(act, Activation) and i % 2 == 0:
                    # no paired brick, append default linear
                    application_methods.append(PenalizedLinear(**kwargs).apply)
                application_methods.append(act.apply)
            else:
                logger.warn("discard activation brick %r due to lack of apply "
                            "mtehod" % str(act))

        super(__class__, self).__init__(
                application_methods, name='_encoder')

        for i in range(1, len(self.children) + 1):
            if hasattr(self.children[-1 * i], 'output_dim'):
                self.output_dim = self.children[-1 * i].output_dim

    @application(input=['x'])
    def apply(self, x):
        return(super(__class__, self).apply(x))


class Decoder(FeedforwardSequence, Initializable):
    """
    decoder brick
    """
    def __init__(self, activations=None, **kwargs):
        """
        arrange and group basic bricks to construct decoder

        Notice: 
        1. If weights_init keyword is not given or given but not an instance of
        NdarrayInitialization, will assume tied-weight used for Decoder and a
        Trasponse Activation brick will be supplied by Decoder automatically.
        However, user must call initialize method of the topmost brick or
        manually push_initialization_config of AutoEncoder brick in order to
        push initialization config from the AutoEncoder
        parent and assign transpose weight from its slibing encoder; 
        2. If weights_init keyword is given and is an instance of
        NdarrayInitialization, then Decoder will use providing initialization
        scheme and allocate shared parameter which is independent of choice of
        encoder weight

        @params activations is a list containing blocks.bricks with apply
        methods
        """
        application_methods = deque(
                act.apply for act in activations if hasattr(act, 'apply'))
        if not isinstance(kwargs.get('weights_init'),
                          initialization.NdarrayInitialization):
            application_methods.appendleft(Transpose(
                kwargs.get('weights_init'), **kwargs).apply)
        else:
            application_methods.appendleft(PenalizedLinear(**kwargs).apply)

        super(__class__, self).__init__(application_methods, name='_decoder')

    def _push_initialization_config(self):
        """
        push the encoder's weight to decoder
        """
        if self.parents:
            selector = select.Selector(self.parents)
            for param_name, param in selector.get_parameters(
                    parameter_name='W').items():
                if param_name.find('_encoder') > 0:
                    setattr(self.children[0], 'ptr_param', param)
                    break

        for i in range(1, len(self.children) + 1):
            if hasattr(self.children[-1 * i], 'output_dim'):
                self.output_dim = self.children[-1 * i].output_dim

    @application(input=['x'])
    def apply(self, x):
        return(super(__class__, self).apply(x))


def proc_data(func):
    """
    module scoped unbound process data deliver training stream
    """
    def wrapper(self, X, y=None, **fit_params):
        # handle the case where X is raw data and need to transformed into
        # DataStream
        if isinstance(X, (DataStream, Transformer)):
            stream = X
            if hasattr(X.iteration_scheme, 'batch_size'):
                batch_size = X.iteration_scheme.batch_size 
                if batch_size != 1 and not isinstance(X, Stacking):
                    stream = Stacking(X, which_sources=('x',))
        else:
            batch_size = fit_params.pop('batch_size', 1)
            stream = Stacking(DataStream(
                IndexableDataset({'x': X}),
                iteration_scheme=SequentialScheme(examples=len(X),
                    batch_size=batch_size)), which_sources=('x',))
        if isinstance(func, theano.compile.function_module.Function):
            # transform
            return(func(x['x'], **fit_params) for x in
                    stream.get_epoch_iterator(as_dict=True))
        else:
            # fit
            return(func(stream, **fit_params))

    if not hasattr(func, '__name__'):
        # will contain class name / included on purpose
        func.__name__ = func.name.split('.')[-1]
        func.__qualname__ = '.'.join([__name__, func.name])
    else:
        func.__qualname__ = '.'.join([__name__, func.__name__])
    func.__module__ = __name__
    return(update_wrapper(wrapper, func))


def _score(self, X, y=None, **kwargs):
    assert(hasattr(self, 'score_samples'))
    return(numpy.mean(list(self.score_samples(X, y=None, **kwargs))))


def patch_algostate_getter(algo_inst):
    """
    adding __getstate__ to ensure return all pickable objects 
    """

    if not hasattr(algo_inst, '_function'):
        logger.warning("{} should have been inistialized".format(algo_inst))
    
    def _getstate(self):
        state = {}
        state['_function'] = algo_inst._function.copy(share_memory=False,
                delete_updates=False, swap=None)
        for name, attr in algo_inst.__dict__.items():    
            if hasattr(state['_function'], name):
                state[name] = getattr(state['_function'], name)
        return state

    return(_getstate)


class RecursiveAutoEncoder(CompilerABC, BaseEstimator, TransformerMixin):
    """
    able to cache the configured instance and return pre-compiled theano
    functions if same parameters are passing to the __init__; othewise,
    re-compile and cache newly created instances
    """
    allocate_args = ('optimizer', 'n_components', 'vocab_size')
    alpha = SharedAccess(0.0) 
    learning_rate = SharedAccess(0.1)

    def __init__(self, optimizer='GradientDescent', n_components=10,
            vocab_size=137, alpha=0.0, n_iter=1, learning_rate=0.01,
            init_weight=None, random_state=None, loss='sqr',
            penalty='l2', fit_intercept=False):

       self.optimizer = optimizer 
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

    def __new__(cls, *args, **kwargs): 
        """
        static method automatically without actually declare it
        """
        # pointing to object.__new__ and cls is mandatory argument should be
        # passed in (equivalently as object.__new__(cls))
        return(super(RecursiveAutoEncoder, cls).__new__(cls))

    def __str__(self):
        assert(hasattr(self, 'n_components'))
        return("%s(dim=%d)"%(self.__class__.__name__, self.n_components))

    def __getstate__(self):
        state = {}
        for name, param in self.__dict__.items():
            if not isinstance(param, types.MethodType):
                state[name] = copy(param)

        for name, param in self.__dict__.items():
            if name not in state:
                if hasattr(param, '__wrapped__'):  # wrapped method
                    wrapped_copy = None
                    if isinstance(param.__wrapped__,
                            theano.compile.function_module.Function):
                        wrapped_copy = param.__wrapped__.copy(share_memory=False,
                            delete_updates=False, swap=None, name='%s'%(
                                param.name))
                    else:
                        wrapped_copy = self.driver(copy(self.optimizer_),
                            variables=[p.copy() for p in
                                inspect.getclosurevars(param.__wrapped__).nonlocals['variables']])
                    assert(wrapped_copy)
                    rewraps_copy = proc_data(wrapped_copy)
                    state[name] = rewraps_copy
                else: # not wrapped method
                    assert(name == 'score')
                    state[name] = _score
        return(state)

    def __setstate__(self, state):
        for name, param in state.items():
            if not isinstance(param, types.FunctionType):
                setattr(self, name, param)
            else:
                setattr(self, name, types.MethodType(param, self))
                
    def set_params(self, **params):
        sk_params = {}
        model_params = self.model.get_parameter_values() 
        for name, val in params.items():
            snames = name.split('__', maxsplit=1)
            if snames[0] == 'model':  # theano parameters
                assert(isinstance(val, dict))
                model_params.update(val)
            else:
                sk_params[name] = val
        self.model.set_parameter_values(model_params) 
        return(super(RecursiveAutoEncoder, self).set_params(**sk_params))
            
    def instantiate(self, *args, **kwargs):
        namespace = kwargs.copy()
        if args:
            optimizer, n_components, vocab_size = args
        else:
            for argname in self.allocate_args:
                namespace[argname] = kwargs.pop(argname)

        compiled_ns = self.compile(namespace['optimizer'],
                namespace['n_components'], namespace['vocab_size'], **kwargs)
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
    

    def compile(self, optimizer, n_components, vocab_size, **kwargs):
        """
        compile theano expressions as functions

        """
        self.alpha = kwargs.pop('alpha')
        self.learning_rate = kwargs.pop('learning_rate') 
        penalty = kwargs.pop('penalty')
        n_iter = kwargs.pop('n_iter')
        fit_intercept = kwargs.pop('fit_intercept')
        activations = kwargs.pop('activations')

        if kwargs.pop('loss') == 'sqr':
            loss_ = SquaredError(name='sqr_cost')
        
        # ensure using the same seed specified by user
        random_generator =  check_random_state(kwargs.pop('random_state')) 
        wc = initialization.Orthogonal().generate(
            random_generator, (2 * n_components, n_components))
        init_coef_ = initialization.Constant(wc)

        autoencoder = AutoEncoder(activations=activations,
                                  dims=n_components,
                                  encoder__weights_init=init_coef_,
                                  encoder__use_bias = fit_intercept,
                                  encoder__penalty = penalty,
                                  decoder__penalty = penalty,
                                  decoder__use_bias = fit_intercept,
                                  cost__brick=loss_.apply)

        rnn = RecursiveBrick(autoencoder, name='recursive_autoencoder')

        we = initialization.Orthogonal(1).generate(
            random_generator, (vocab_size + 1, n_components))

        embed = kwargs.pop('init_weight', None)
        if isinstance(embed, numpy.ndarray):
            we[:-1, :] = embed 

        embedding = LookupTree(dim=n_components,
            length=vocab_size + 1, name='embed')
        embedding.weights_init = initialization.Constant(we)

        brick = TreeBrickWrapper([embedding.apply, rnn.apply],
                name='treeop_wrapper')

        x = theano.sparse.csr_matrix('x', dtype='int32')
        add_role(x, blocks.roles.INPUT)
        
        if not brick.initialized:
            brick.initialize()

        self.__model = Model(brick.apply(x,
            'error', decode__unfold=True, encode__reverse=True))
        assert(len(self.__model.outputs) == 3)

        aux = [v for v in self.__model.auxiliary_variables if
               'norm' in v.name and v.name.index('norm') > 0]

        if self.alpha > 0.:
            penalized_loss = self.__model.outputs[-1] + (self.alpha *
                    tensor.ones(len(aux), dtype=theano.config.floatX) *
                    tensor.stack(aux, axis=0)).sum()
            penalized_loss.name = 'penalized_square_error'
            self.__model.outputs.append(penalized_loss)

        # TreeOp only does transformation not differentiable
        indifferentiables = VariableFilter(
                roles=[blocks.roles.INPUT],
                bricks=[brick])(self.__model.variables)
        parameters = VariableFilter(
                roles=[blocks.roles.PARAMETER])(self.__model.variables)
        if type(optimizer) is str:
            optimizer = getattr(blocks.algorithms, optimizer)
        optimizer_ = optimizer(cost=self.__model.outputs[-1],
                parameters=parameters,
                step_rule=Scale(learning_rate=self.learning_rate),
                consider_constant=indifferentiables)
        compiled_ns = {}

        # encoding
        compiled_ns['transform'] = proc_data(theano.function([x],
            self.__model.outputs[0], name="%s.%s"%(self.__class__.__name__,
                'transform')))
        # decoding
        compiled_ns['inverse_transform'] = proc_data(theano.function([x],
            self.__model.outputs[1], name="%s.%s"%(self.__class__.__name__,
                'inverse_transform')))
        # error
        compiled_ns['score_samples'] = proc_data(theano.function([x],
            self.__model.outputs[-1], name="%s.%s"%(self.__class__.__name__,
                'score_samples')))
        # fit
        compiled_ns['fit'] = proc_data(self.driver(optimizer_,
                variables=[self.__model.outputs[-1]] + aux))
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
        main_loop = partial(MainLoop, algorithm=optimizer_,
                model=self.__model)

        @wraps(MainLoop.run)
        def loop_wrapper(training_stream, **fit_params):
            backend = fit_params.get('log_backend')
            save_models = fit_params.get('save_model', True)
            default_extensions = [
                    TrainingDataMonitoring(variables=variables,
                                           prefix=str(self),
                                           after_batch=True),
                    FinishAfter(after_n_epochs=self.n_iter),
                    Timestamp(), Timing(), Printing()]
            if save_models: 
                default_extensions += [
                        saveload.Checkpoint(
                            os.path.join(data_dir, '%s.model' % (str(self))),
                    save_separately=None, use_cpickle=False),
                        saveload.Load(
                            os.path.join(data_dir, '%s.model' % (str(self))),
                    load_iteration_state=False, load_log=False)]
            logger.info('start unsupervised pre-training step ({} with '
                    'batch_size = {}'.format(str(self),
                    training_stream.data_stream.iteration_scheme.batch_size))
            main_loop(data_stream=training_stream, 
                extensions=default_extensions,
                log_backend=backend).run()
            logger.info('complete unsupervised pre-training step %s'%
                str(self))
            return(self)

        return(loop_wrapper)
