import os
import sys
import time
import numpy
import theano

from .recursnn_rae import AbstractAutoEncoder
from .recursnn_rae import split_params
from .recursnn_rbm import load_streams
from .recursnn_rbm import PlotExtension

from symlearn.blocks.bricks import Bias
from symlearn.blocks import initialization
from symlearn.blocks import graph
from symlearn.blocks.bricks.cost import BinaryCrossEntropy
from symlearn.blocks.bricks.cost import SquaredError
from theano import tensor
from functools import partial


# start-snippet-1
class DenoiseAutoEncoder(AbstractAutoEncoder):
    """
    a block version from modified Denoising Auto-Encoder class (dA) on the
    DeepLearningTutorial website
    """

    def __init__(self, dims, activations, **kwargs):
        """
        W is initialized with `initial_W` which is uniformely sampled
        from -4*sqrt(6./(n_visible+n_hidden)) and
        4*sqrt(6./(n_hidden+n_visible))the output of uniform if
        converted using asarray to dtype
        theano.config.floatX so that the code is runable on GPU
        """

        n_visible, n_hidden = dims
        # setting encoder activations
        params = kwargs.copy()
        params['encoder__activations'] = activations
        params['encoder__input_dim'] = n_visible
        params['encoder__output_dim'] = n_hidden
        super(__class__, self).__init__(**params)

    def _process_params(self, **params):

        init_params = split_params(**params)

        # setting default params for encoder
        if init_params['encoder']['weights_init'] is None:
            low = -4 * numpy.sqrt(6. / (init_params['encoder']['input_dim'] +
                init_params['encoder']['output_dim']))
            high = 4 * numpy.sqrt(6. / (init_params['encoder']['input_dim'] +
                init_params['encoder']['output_dim']))
            init_params['encoder']['weights_init'] = initialization.Uniform(
                    numpy.mean([low, high]), width=(high - low) / 2,
                    dtype=theano.config.floatX)

        # setting default params for decoder
        init_params['decoder']['input_dim'] = \
                init_params['encoder']['output_dim']
        init_params['decoder']['output_dim'] = \
                init_params['encoder']['input_dim']
        if init_params['decoder'].get('use_bias'):
            extra_bricks = [Bias(init_params['decoder']['output_dim'],
                biases_init=initialization.Constant(
                    numpy.zeros(init_params['decoder']['output_dim'])),
                name='b_hidden')]
        else:
            extra_bricks = []

        init_params['decoder']['activations'] = \
            init_params['encoder']['activations'] + extra_bricks
        return(init_params, [])

    def _encode(self, x, corruption_level=0., noise='dropout'):
        if noise == 'dropout':  # no scale
            apply_func = partial(graph.apply_dropout, custom_divisor=1.)
        elif noise == 'gaussian':
            apply_func = graph.apply_noise
        y = self.enc_proxy.apply(x)
        cg = graph.ComputationGraph(y)
        return(apply_func(cg, cg.inputs, corruption_level).outputs.pop())

    def _decode(self, y, epsilon=1e-5):
        return((self.dec_proxy.apply(y).clip(epsilon, 1 - epsilon)))

    def _reconstruct_cost(self, x, corruption_level=0., noise='dropout'):
        y = self._encode(x, corruption_level, noise)
        z = self._decode(y)

        if noise == 'dropout':  # prefer input x within [0, 1]
            return(tensor.mean(BinaryCrossEntropy(name='binary_cost').apply(
                tensor.shape_padleft(x), tensor.shape_padleft(z))))
        elif noise == 'gaussian':  # prefer input x is continuous real number
            return(tensor.mean(SquaredError(name='sqaure_cost').apply(
                tensor.shape_padleft(x), tensor.shape_padleft(z))))


def test_dA(learning_rate=0.1, training_epochs=15, n_samples=10,
            batch_size=20, output_folder='dA_plots', corruption_level=0.):

    """
    This demo is tested on MNIST

    :type learning_rate: float
    :param learning_rate: learning rate used for training the DeNosing
                          AutoEncoder

    :type training_epochs: int
    :param training_epochs: number of epochs used for training

    :type dataset: string
    :param dataset: path to the picked dataset

    """
    from symlearn.blocks import model
    from symlearn.blocks.main_loop import MainLoop
    from symlearn.blocks.algorithms import GradientDescent
    from symlearn.blocks.extensions import FinishAfter, Printing, Timing
    from symlearn.blocks.extensions.monitoring import TrainingDataMonitoring
    from symlearn.blocks.bricks import Logistics

    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)

    ####################################
    # BUILDING THE MODEL NO CORRUPTION #
    ####################################

    da = DenoiseAutoEncoder(dims=[28 * 28, 500], activations=[Logistics()],
                encoder__weights_init=initialization.Orthogonal(),
                encoder__use_bias=False, decoder__use_bias=False)
    x = tensor.matrix('x')  # the data is presented as rasterized images
    cost = da.apply(x, corruption_level=corruption_level)
    da_model = model.Model(cost)
    extentions = [TrainingDataMonitoring(cost), PlotExtension(output_folder),
                Timing(), Printing(),
                FinishAfter(after_n_epochs=training_epochs)]
    datastreams = load_streams(n_samples, batch_size)
    sgd = GradientDescent(cost, da_model.get_parameter_values())
    main_loop = MainLoop(sgd, datastreams[0], da_model, extentions=extentions)

    start_time = time.clock()
    ############
    # TRAINING #
    ############

    main_loop.run()
    end_time = time.clock()
    training_time = (end_time - start_time)

    print('The %d \% corruption code for file ' + os.path.split(__file__)[1] +
            ' ran for %.2fm' % (corruption_level, (training_time / 60.)),
            file=sys.stderr)

if __name__ == '__main__':
    test_dA(corruption_level=0.3)
