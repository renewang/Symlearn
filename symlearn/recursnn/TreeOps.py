from collections import OrderedDict
from theano.gof.null_type import NullType
from theano.gradient import DisconnectedType
from theano import tensor

from . import recursnn_helper
from . import recursnn_utils

import theano
import numpy

class ExtractTreeFeatures(theano.Op):
    __prop__ = () # used to compute hash 

    def __init__(self, vocab_size=0):
        self.vocab_size = vocab_size
        super(__class__, self).__init__()

    def make_node(self, x, embed):
        """
        make_node will accomplish three things:
        1. ensure the inputs are symbolic theano tensor variables of valid
        types
        2. infer the symbolic theano tensor variables from inputs if possible
        3. to bind the symbolic outputs as Apply instance' attributes and
        return the bound Apply instance

        return:

        """
        x = theano.sparse.as_sparse_or_tensor_variable(x,
                name=getattr(x, 'name'))
        embed = tensor.as_tensor_variable(embed,
                name=getattr(embed, 'name'))
        intp = numpy.dtype(numpy.intp)      # index type
        return(theano.Apply(op=self, inputs=[x, embed],
            outputs=[
                tensor.matrix(name='traversal_order', dtype=intp),
                tensor.tensor3(name='indices', dtype='uint8'),
                tensor.tensor3(name='span_range', dtype=intp),
                tensor.tensor3(name='mapping', dtype=intp),
                tensor.matrix(name='mask', dtype='int8')]))

    def perform(self, node, inputs, output_storage):
        """
        here is python implementations with real value not symbolical
        expression

        notice: don't conduct any normalization in this phase, leave this to
        training model
        """
        x, embed = inputs
        assert(x.format == 'csr')
        if self.vocab_size == 0:
            self.vocab_size = embed.shape[0]
        outs = output_storage
        assert(x.shape[0] % x.shape[1] == 0)
        n_batch = x.shape[0] // x.shape[1]

        batch_results = OrderedDict()
        for outputvar in node.outputs:
            batch_results.setdefault(outputvar.name, [])

        # collect batch results and compute the max_traversal_len
        max_len = -1
        for i in range(n_batch):
            start = i * x.shape[1]
            end = (i + 1) * x.shape[1]
            result = recursnn_helper._iter_matrix_groups(x[start:end],
                    vocab_size=self.vocab_size,
                    traverse_func=recursnn_utils.two_passes_traversal)

            if result[1].shape[0] > max_len:
                max_len = result[1].shape[0]

            for name, res in zip(batch_results.keys(), result):
                if name == 'indices':
                    res = res.toarray()
                batch_results[name].append(res)

        max_len += 1  # for dummy padding
        assert(max_len > 0)

        # creating mask for different tree size
        # mask array should be (the_max_traversal_order + 1) x n_batch
        batch_results['mask'] = []
        for split in batch_results['indices']:
            batch_results['mask'].append(
                    numpy.zeros(max_len, dtype=numpy.int8))
            batch_results['mask'][-1][len(split):] = True

        # padding
        for name in frozenset(batch_results.keys()):
            const_val, pad_len = 0, max_len
            if name == 'mapping' or name == 'traversal_order':
                const_val = max_len - 1  # will refer to the dummy padding
                if name == 'traversal_order':
                    pad_len = max_len // 2 - 1
            for i, split in enumerate(batch_results[name][:]):
                assert(len(split) <= pad_len)
                if len(split) != pad_len:
                    pad_width = ((0, pad_len - len(split)), (0, 0),)
                    batch_results[name][i] = numpy.pad(batch_results[name][i],
                                pad_width[:split.ndim], mode='constant',
                                constant_values=const_val)

        # expanding and then stacking
        for name in batch_results.keys():
            batch_results[name] = numpy.concatenate([numpy.expand_dims(e,
                axis=1) for e in batch_results[name]], axis=1)

        # ensure the dtype is comforming output
        for outputvar in node.outputs:
            batch_results[outputvar.name] = batch_results[
                    outputvar.name].astype(outputvar.type.numpy_dtype)

        for i, out_val in enumerate(batch_results.values()):
            outs[i][0] = out_val

    def grad(self, inputs, output_grads):
        """
        compute gradient with respect to embed parameter

        @param inputs is a list of Tensor variables of Op's input (not much use
        here) 
        @param output_grads is a list of Tensor variables precomputed within
        theano.gradient.grad method whose values it the gradient of cost with
        respect to Op's output. The length should equal to the length of Op's
        output and in the order of make_node output: 
        [traversal_order, enc_mat, span_range, mapping, mask]
        only enc_mat can be differentiaed with respect to embed
        """
        x, embed = inputs
        return [theano.gradient.grad_undefined(self, 0, x,
            comment="cannot take gradient with respect to graph itself"
                    "which is not parameter"), 
                theano.gradient.grad_undefined(self, 0, embed,
            comment="cannot take gradient over graph with respect to embed")]
