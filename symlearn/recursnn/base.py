from theano import tensor, shared

from functools import singledispatch, partial
from abc import abstractmethod, ABCMeta
from contextlib import contextmanager
from copy import copy

from symlearn.blocks.bricks import Logistic

import weakref
import inspect
import theano
import time
import logging

logger = logging.getLogger(__name__)

@contextmanager
def timethis(func, n_iter):
    start = time.time()
    logger.info('{} start training for total {:d} '
            'iterations'.format(func.__name__, n_iter))
    yield func()
    end = time.time()
    logger.info('training {} takes {:.3f} seconds for '
            'total {:d} iterations'.format(func.__name__,
                    end - start, n_iter))
    
class SharedAccess(object):
    """
    descriptors used for theano shared variable
    """

    def __init__(self, initval=0., name=None):
        self.val = shared(initval)
        self.name = name

    def __get__(self, obj, objtype): 
        val = self.val.get_value(borrow=True)
        if val.ndim == 0: # for scalar return python native type
            val = val.item()
        return val 
   
    def __set__(self, obj, val):
        return self.val.set_value(val, borrow=False)

        
class CompilerABCMeta(type):
    def __init__(mcls, name, bases, namespace, *args, **kwargs):
        """
        customize the initialization of metaclass in order to gain the
        advantages of different class creation methods
        """
        super(__class__, mcls).__init__(name, bases, namespace)
        mcls.__cache = weakref.WeakValueDictionary()

    def __new__(mcls, clsname, bases, namespace):
        for name, value in namespace.items():
            if isinstance(value, SharedAccess):
                value.name = name
                namespace[name] = value
        return(type.__new__(mcls, clsname, bases, namespace))

    def __call__(mcls, *args, **kwargs):
        # create an empty object
        inst = super(__class__, mcls).__call__(*args, **kwargs)
        # get the default values for constructor 
        params = inst.get_params()
        allocate_args = ()
        if args:
            allocate_args += args # specify required args
            params.update({argname: arg for arg, argname in zip(args,
                mcls.allocate_args)})
        elif kwargs:
            allocate_args = tuple([kwargs.get(argname, params[argname]) for
                argname in mcls.allocate_args]) # specify required args via
                                                # keywords
            params.update(kwargs)
        else:
            allocate_args = tuple([params[argname] for argname in
                mcls.allocate_args]) # construct default args

        if allocate_args in mcls.__cache:
            orig_inst = mcls.__cache[allocate_args] # get original
            inst = copy(orig_inst) # should return weakref.ref(inst)
        else:
            if 'activations' in kwargs:
                activations = kwargs.pop('activations')
            else:
                activations = [Logistic(name='sigmoid')]
            inst.instantiate(activations=activations, **params)
            mcls.__cache[allocate_args] = inst 
        return(inst)


class CompilerABC(metaclass=CompilerABCMeta):

    @abstractmethod
    def instantiate(cls, clsns=None, bases=None, clskwargs={}):
        """
        used to populate class namespace of class will be created dynamically
        through this method with the functions compiled by the compile methods 
       
        @param clsns is a dict which stores customized class namespace
        @param bases is a tuple and used to specified base classes used to
            dynamically create class
        @param clskwargs is a class dict used to passed to callback to update
            the class namespace
        """
        pass

    @abstractmethod
    def compile(cls, optimizer, *args, **kwargs):
        """
        abstract method to compile function with specific optimizer 
        """
        pass

    @abstractmethod
    def driver(cls, optimizer_, **kwargs): 
        """
        return a MainLoop instance prebound to optimizer built through compile
        method and only leaving data_stream for the client code
        """
        pass

@singledispatch
def get_phrases_helper(first, trees, vocab=None):
    raise ValueError("fails to find the corresponding type {!r}".format(
        first.__class__))


def _hash_phrase(phrase, hash_func=None, seed=0):
    """
    hash phrase with the provided hash_func choice. default is
    identity which is no-transform)
    """
    lookup = str.maketrans({'(': '-LRB-', ')': '-RRB-'})

    def identity(phrase):
        if type(phrase) == str:
            return(phrase.translate(lookup))
        elif type(phrase) == bytes:
            return(phrase.decode('utf-8').translate(lookup))
    if hash_func is None:
        hash_func = identity
    return(hash_func(phrase))


def split_params(**params):
    """
    a scikit-learn way to pass parameters to different sub-components, maybe
    call functions in scikit-learn suffices
    """
    init_params = dict()
    for pname, pval in params.items():
        if pname.find('__') > 0:
            step, param = pname.split('__', 1)
            init_params.setdefault(step, {})
            init_params[step][param] = pval
        else:
            # push back the original parameter names and values
            init_params[pname] = pval
    return(init_params)


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