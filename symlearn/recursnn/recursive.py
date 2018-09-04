from symlearn.blocks.bricks.recurrent import BaseRecurrent, recurrent
from symlearn.blocks.bricks.wrappers import BrickWrapper
from symlearn.blocks.bricks import (Identity, Sequence, Softmax, MLP, Linear)
from symlearn.blocks.bricks.base import (application, rename_function, lazy)
from symlearn.blocks.bricks.lookup import LookupTable
from symlearn.blocks.utils import (dict_subset, check_theano_variable, shared_floatx_nans)
from symlearn.blocks.roles import add_role, WEIGHT, BIAS
from theano import tensor
from collections import OrderedDict

from . import (split_params, inspect_and_bind)
from .TreeOps import ExtractTreeFeatures


import logging
import theano

logger = logging.getLogger(__name__)

class WithTreeOp(BrickWrapper):
    """
    a blocks.BrickWrapper to convert adjacent matrix representing graph
    (parsing tree for instance)
    """
    def wrap(self, wrapped, namespace):
        """
        implment wrap to follow blocks.BrickWrapper contract
        """
        def apply(self, application, *args, **kwargs):
            new_kwargs = kwargs.copy()
            tree_op = new_kwargs.pop('operator', ExtractTreeFeatures())
            # map application inputs to tree_op inputs
            inputs = dict(zip(application.inputs, args))
            if 'embed' not in inputs and isinstance(self.children[0],
                    LookupTable):
                inputs.update({'embed': self.children[0].W})
            bound_args = inspect_and_bind(tree_op.make_node, **inputs)
            op_outputs = tree_op(*bound_args.args)
            # turn the tree_op outputs into kwargs
            wrapped_kwargs = split_params(**new_kwargs)
            for var in op_outputs:
                wrapped_kwargs.setdefault(application.name,
                        OrderedDict())[var.name] = var
            wrapped_args = args
            outputs = wrapped.__get__(self, None)(*wrapped_args,
                        **wrapped_kwargs.get(application.name))
            return(outputs)

        def apply_delegate(self):
            return wrapped.__get__(self, None)

        apply = application(rename_function(apply, wrapped.application_name))

        apply_delegate = apply.delegate(
            rename_function(apply_delegate,
                            wrapped.application_name + "_delegate"))
        namespace[wrapped.application_name] = apply
        namespace[wrapped.application_name + "_delegate"] = apply_delegate


class RecursiveBrick(BaseRecurrent):

    def __init__(self, autoencoder, **kwargs):
        super(__class__, self).__init__(**kwargs)
        self.children = [autoencoder]

    @staticmethod
    def _compute_weight(span_range, mapping, mask):
        """
        helper function to compute the span weight used for cost computation
        """
        if span_range.ndim  == 2:
            span_range = tensor.shape_padaxis(span_range, axis=1)

        batch_idx = tensor.arange(span_range.shape[1])
        word_spans = tensor.stack([
            (span_range[mapping[:, :, 0], batch_idx, 1] -
            span_range[mapping[:, :, 0], batch_idx, 0]),
            (span_range[mapping[:, :, 1], batch_idx, 1] -
                span_range[mapping[:, :, 1], batch_idx, 0])], axis=-1)
        span_weight = tensor.zeros(word_spans.shape,
                dtype=theano.config.floatX)
        span_weight = tensor.set_subtensor(span_weight[(1 - mask).nonzero()],
                (word_spans[(1 - mask).nonzero()] /
                word_spans[(1 - mask).nonzero()].sum(axis=-1, keepdims=True)))
        return(span_weight)

    @application(target=['target'])
    def apply(self, target, *args, **kwargs):
        """
        dispatch which recurrent application should be called
        """
        target_func = getattr(self, target)
        self.apply.target = target_func
        if hasattr(self.apply.target, 'states'):
            self.initial_states.outputs = self.apply.target.states
        return(target_func(*args, **kwargs))

    @application
    def initial_states(self, batch_size, *args, **kwargs):
        result = []
        for state in self.apply.target.states:
            dim = self.get_dim(state)
            if dim == 0:
                result.append(tensor.zeros((batch_size,)))
            else:
                result.append(kwargs.get(state).copy())
        return result

    @recurrent(sequences=['traversal_order'], states=['enc_mat'],
            outputs=['enc_mat'], contexts=['mapping'])
    def encode(self, traversal_order, enc_mat, mapping):
        child_input = (traversal_order, enc_mat, mapping)
        enc_mat = self.children[0].encode_apply(*child_input)
        return(enc_mat)

    @recurrent(sequences=['traversal_order'], states=['dec_mat'],
            outputs=['dec_mat'], contexts=['enc_mat', 'mapping'])
    def decode(self, traversal_order, dec_mat, mapping, enc_mat=None,
            unfold=True):
        if unfold:
            child_input = (traversal_order, dec_mat, mapping)
        else:
            child_input = (traversal_order, dec_mat, mapping, enc_mat)
        dec_mat = self.children[0].decode_apply(*child_input)
        return(dec_mat)

    @recurrent(sequences=['traversal_order'], states=['enc_mat', 'dec_mat'],
            outputs=['enc_mat', 'dec_mat', 'cost'], contexts=['span_weight',
                'mapping', 'mask'])
    def step_cost(self, traversal_order, enc_mat, dec_mat, span_weight,
            mapping, mask):
        enc_mat = self.apply('encode', traversal_order, enc_mat, mapping,
                iterate=False)
        dec_mat = self.apply('decode', traversal_order, dec_mat, mapping,
                enc_mat, unfold=False, iterate=False)
        cost = self.children[0]._reconstruct_cost(traversal_order, enc_mat,
                dec_mat, span_weight, mapping, mask)
        return(enc_mat, dec_mat, cost)

    def error(self, *args, **kwargs):
        """
        compute recursive unfolding or direct cost by calling encoding and
        decoding iteratively
        """
        new_kwargs = split_params(**kwargs)
        weigh_argspec = inspect_and_bind(self._compute_weight, **new_kwargs)
        span_weight = self._compute_weight(*weigh_argspec.args,
                               **weigh_argspec.kwargs)
        new_kwargs['span_weight'] = span_weight
        is_unfold = new_kwargs.get('decode', {}).get('unfold', True)
        if not is_unfold:
            dec_mat = tensor.zeros_like(new_kwargs.get('enc_mat'))
            new_kwargs.update({'dec_mat': dec_mat})
            cost_argspec = inspect_and_bind(
                    self._step_cost, **new_kwargs)
            enc_mat, dec_mat, cost = self.apply('step_cost',
                    *cost_argspec.args, **new_kwargs.get('step_cost', {}))
        else:
            encode_argspec = inspect_and_bind(self._encode, **new_kwargs)
            enc_mat = self.apply('encode', *encode_argspec.args,
                **new_kwargs.get('encode', {}))
            dec_mat = tensor.concatenate([
                tensor.shape_padleft(enc_mat[-1][0]),
                tensor.zeros_like(enc_mat[-1][1:])], axis=0)
            new_kwargs.update({'dec_mat': dec_mat})
            decode_argspec = inspect_and_bind(self._decode,
                        **new_kwargs)
            dec_mat = self.apply('decode', *decode_argspec.args,
                **new_kwargs.get('decode', {}))
            new_kwargs.update({'dec_mat': dec_mat[-1],
                'enc_mat': enc_mat[-1]})
            cost_argspec = inspect_and_bind(self.children[0]._unfold_cost,
                    **new_kwargs)
            cost = self.children[0].apply(*cost_argspec.args,
                    **cost_argspec.kwargs)
        cost.name = 'reconstruct_cost'
        return(enc_mat[-1], dec_mat[-1], cost)

    def get_dim(self, name):
        """
        return dim of states which is required in recurrent
        """
        if name in ['cost', 'mask']:
            return(0)
        elif name in ['span_weight', 'span_range', 'mapping']:
            return(2)  # should return actual branch factor
        elif name in ['enc_mat', 'dec_mat', 'word_mat']:
            # return embedding dim
            return(self.children[0].enc_proxy.input_dim)


class LookupTree(LookupTable):

    # TODO: need some handle with unseen words during training
    @lazy(allocation=['length', 'dim'])
    def __init__(self, length, dim, **kwargs):
        self.penalty = kwargs.get('penalty', 'l2')
        super(__class__, self).__init__(length, dim, **kwargs)

    @application(inputs=['index_mat'], outputs=['enc_mat'])
    def apply(self, index_mat):
        word_mat = tensor.tensordot(index_mat, self.W, axes=[2, 0])
        masking = tensor.shape_padright(1 - (index_mat.sum(axis=-1) > 1))
        enc_mat = word_mat * masking
        enc_mat.name = 'enc_mat'
        return(enc_mat)

    def _allocate(self):
        super(__class__, self)._allocate()
        L = int(self.penalty[-1])
        assert(L == 1 or L == 2)
        self.add_auxiliary_variable(self.W.norm(L),
                name=self.name + '__W_norm__'+ self.penalty)

class PenalizedLinear(Linear):

    @lazy(allocation=['input_dim', 'output_dim'])
    def __init__(self, input_dim, output_dim, **kwargs):
        self.penalty = kwargs.pop('penalty', 'l2')
        super(__class__, self).__init__(input_dim, output_dim, **kwargs)


    def _allocate(self):
        L = int(self.penalty[-1])
        assert(L == 1 or L == 2)
        W = shared_floatx_nans((self.input_dim, self.output_dim), name='W')
        add_role(W, WEIGHT)
        self.parameters.append(W)
        self.add_auxiliary_variable(W.norm(L),
                name=self.parents[0].name + '__W_norm__'+ self.penalty)
        if getattr(self, 'use_bias', True):
            b = shared_floatx_nans((self.output_dim,), name='b')
            add_role(b, BIAS)
            self.parameters.append(b)
            self.add_auxiliary_variable(b.norm(L),
                    name=self.parents[0].name + '__b_norm__'+ self.penalty)

class TreeBrick(Sequence):
    def __init__(self, applications, **kwargs):
        super(__class__, self).__init__(applications, **kwargs)

    @application(inputs=['x'], outputs=['enc', 'dec', 'cost'])
    def apply(self, application_call, x, *args, **kwargs):
        inputs = args
        newkws = kwargs.copy()
        for i, child in enumerate(self.children):
            try:
                extra_arg =  newkws.pop('indices')
                outputs = child.apply(extra_arg)
                newkws.update({'enc_mat': outputs})
            except KeyError as e:
                outputs = child.apply(*inputs, **newkws)
                inputs = outputs
        return(outputs)


class TreeBrickWrapper(TreeBrick):
    decorators = [WithTreeOp()]
