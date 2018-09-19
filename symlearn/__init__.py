import theano
import theano.sparse


def _copy(sp_var):
    clone_args = tuple(d.copy() for d in theano.sparse.csm_properties(sp_var))
    return(theano.sparse.CSR(*clone_args))


setattr(theano.sparse.SparseVariable, 'copy', _copy)

sparse = theano.sparse