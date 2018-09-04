"""This tutorial introduces restricted boltzmann machines (RBM) using Theano.

Boltzmann Machines (BMs) are a particular form of energy-based model which
contain hidden variables. Restricted Boltzmann Machines further restrict BMs
to those without visible-visible and hidden-hidden connections.
"""
import time
import numpy

import theano
import theano.tensor as T
import os

from itertools import chain
from symlearn.fuel.datasets import MNIST
from symlearn.fuel.streams import DataStream
from symlearn.fuel.schemes import SequentialScheme
from symlearn.fuel.datasets import IterableDataset
from symlearn.fuel.transformers import Flatten

from utils import tile_raster_images
from collections import OrderedDict
from symlearn.blocks.extensions import SimpleExtension
from symlearn.blocks import initialization
from symlearn.blocks.bricks import Bias
from symlearn.blocks import select

from recursnn.recursnn_rae import AbstractAutoEncoder
from recursnn.recursnn_rae import split_params

try:
    import PIL.Image as Image
except ImportError:
    import Image


def PlotImages(n_samples, n_chains, sample_fn, output_folder='rbm_plots'):
    # create a space to store the image for plotting ( we need to leave
    # room for the tile_spacing as well)
    image_data = numpy.zeros(
        (29 * n_samples + 1, 29 * n_chains - 1),
        dtype='uint8'
    )
    for idx in range(n_samples):
        # generate `plot_every` intermediate samples that we discard,
        # because successive samples in the chain are too correlated
        vis_mf, vis_sample = sample_fn()
        print(' ... plotting sample ', idx)
        image_data[29 * idx:29 * idx + 28, :] = tile_raster_images(
            X=vis_mf,
            img_shape=(28, 28),
            tile_shape=(1, n_chains),
            tile_spacing=(1, 1)
        )

    # construct image
    image = Image.fromarray(image_data)
    image.save(os.path.join(output_folder, 'samples.png'))


class PlotExtension(SimpleExtension):
    def __init__(self, output_folder, **kwargs):
        kwargs.setdefault("after_epoch", True)
        self.output_folder = output_folder
        self.plotting_time = 0.
        super(__class__, self).__init__(**kwargs)

    def do(self, callback_name, *args):
        """Write the values of monitored variables to the log."""
        # Plot filters after each training epoch
        plotting_start = time.clock()
        # Construct image from the weight matrix
        image = Image.fromarray(
            tile_raster_images(
                X=self.main_loop.model._parameter_dict['W'].get_value(
                    borrow=True).T,
                img_shape=(28, 28),
                tile_shape=(10, 10),
                tile_spacing=(1, 1)
            )
        )
        image.save(os.path.join(self.output_folder, 'filters_at_epoch_%i.png' %
            self.main_loop.log['epochs_done']))
        plotting_stop = time.clock()
        self.plotting_time += (plotting_stop - plotting_start)


# start-snippet-1
class RBM(AbstractAutoEncoder):
    """
    a block version from modified Restricted Boltzmann Machine (RBM) on the
    DeepLearningTutorial website
    """
    def __init__(self, dims, activations, **kwargs):
        n_visible, n_hidden = dims
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
        extra_bricks = [Bias(init_params['decoder']['output_dim'],
            biases_init=initialization.Constant(
                numpy.zeros(init_params['decoder']['output_dim'])),
            name='vbias')]

        extra_bricks += []

        init_params['decoder']['activations'] = \
            init_params['encoder']['activations'] + extra_bricks
        return(init_params, [])

    def free_energy(self, v_sample):
        ''' Function to compute the free energy '''
        wx_b = self.enc_proxy.apply(v_sample)
        brick_select = select.Selector(self.dec_proxy)
        params = brick_select.get_parameters()
        vbias = params['vbias']
        vbias_term = T.dot(vbias, v_sample)
        hidden_term = T.sum(T.log(1 + T.exp(wx_b)), axis=1)
        return -hidden_term - vbias_term

    def propup(self, vis):
        '''This function propagates the visible units activation upwards to
        the hidden units

        Note that we return also the pre-sigmoid activation of the
        layer. As it will turn out later, due to how Theano deals with
        optimizations, this symbolic variable will be needed to write
        down a more stable computational graph (see details in the
        reconstruction cost function)

        '''
        pre_sigmoid_activation = self.enc_proxy.apply(vis)
        return [pre_sigmoid_activation, T.nnet.sigmoid(pre_sigmoid_activation)]

    def sample_h_given_v(self, v0_sample):
        ''' This function infers state of hidden units given visible units '''
        # compute the activation of the hidden units given a sample of
        # the visibles
        pre_sigmoid_h1, h1_mean = self.propup(v0_sample)
        # get a sample of the hiddens given their activation
        # Note that theano_rng.binomial returns a symbolic sample of dtype
        # int64 by default. If we want to keep our computations in floatX
        # for the GPU we need to specify to return the dtype floatX
        h1_sample = self.theano_rng.binomial(size=h1_mean.shape,
                                             n=1, p=h1_mean,
                                             dtype=theano.config.floatX)
        return [pre_sigmoid_h1, h1_mean, h1_sample]

    def propdown(self, hid):
        '''This function propagates the hidden units activation downwards to
        the visible units

        Note that we return also the pre_sigmoid_activation of the
        layer. As it will turn out later, due to how Theano deals with
        optimizations, this symbolic variable will be needed to write
        down a more stable computational graph (see details in the
        reconstruction cost function)

        '''
        pre_sigmoid_activation = self.dec_proxy.apply(hid)
        return [pre_sigmoid_activation, T.nnet.sigmoid(pre_sigmoid_activation)]

    def sample_v_given_h(self, h0_sample):
        ''' This function infers state of visible units given hidden units '''
        # compute the activation of the visible given the hidden sample
        pre_sigmoid_v1, v1_mean = self.propdown(h0_sample)
        # get a sample of the visible given their activation
        # Note that theano_rng.binomial returns a symbolic sample of dtype
        # int64 by default. If we want to keep our computations in floatX
        # for the GPU we need to specify to return the dtype floatX
        v1_sample = self.theano_rng.binomial(size=v1_mean.shape,
                                             n=1, p=v1_mean,
                                             dtype=theano.config.floatX)
        return [pre_sigmoid_v1, v1_mean, v1_sample]

    def gibbs_hvh(self, h0_sample):
        ''' This function implements one step of Gibbs sampling,
            starting from the hidden state'''
        pre_sigmoid_v1, v1_mean, v1_sample = self.sample_v_given_h(h0_sample)
        pre_sigmoid_h1, h1_mean, h1_sample = self.sample_h_given_v(v1_sample)
        return [pre_sigmoid_v1, v1_mean, v1_sample,
                pre_sigmoid_h1, h1_mean, h1_sample]

    def gibbs_vhv(self, v0_sample):
        ''' This function implements one step of Gibbs sampling,
            starting from the visible state'''
        pre_sigmoid_h1, h1_mean, h1_sample = self.sample_h_given_v(v0_sample)
        pre_sigmoid_v1, v1_mean, v1_sample = self.sample_v_given_h(h1_sample)
        return [pre_sigmoid_h1, h1_mean, h1_sample,
                pre_sigmoid_v1, v1_mean, v1_sample]

    # start-snippet-2
    def get_cost_updates(self, lr=0.1, persistent=None, k=1):
        """This functions implements one step of CD-k or PCD-k

        :param lr: learning rate used to train the RBM

        :param persistent: None for CD. For PCD, shared variable
            containing old state of Gibbs chain. This must be a shared
            variable of size (batch size, number of hidden units).

        :param k: number of Gibbs steps to do in CD-k/PCD-k

        Returns a proxy for the cost and the updates dictionary. The
        dictionary contains the update rules for weights and biases but
        also an update of the shared variable used to store the persistent
        chain, if one is used.

        """

        # compute positive phase
        pre_sigmoid_ph, ph_mean, ph_sample = self.sample_h_given_v(self.input)

        # decide how to initialize persistent chain:
        # for CD, we use the newly generate hidden sample
        # for PCD, we initialize from the old state of the chain
        if persistent is None:
            chain_start = ph_sample
        else:
            chain_start = persistent
        # end-snippet-2
        # perform actual negative phase
        # in order to implement CD-k/PCD-k we need to scan over the
        # function that implements one gibbs step k times.
        # Read Theano tutorial on scan for more information :
        # http://deeplearning.net/software/theano/library/scan.html
        # the scan will return the entire Gibbs chain
        (
            [
                pre_sigmoid_nvs,
                nv_means,
                nv_samples,
                pre_sigmoid_nhs,
                nh_means,
                nh_samples
            ],
            updates
        ) = theano.scan(
            self.gibbs_hvh,
            # the None are place holders, saying that
            # chain_start is the initial state corresponding to the
            # 6th output
            outputs_info=[None, None, None, None, None, chain_start],
            n_steps=k
        )
        # start-snippet-3
        # determine gradients on RBM parameters
        # note that we only need the sample at the end of the chain
        chain_end = nv_samples[-1]

        cost = T.mean(self.free_energy(self.input)) - T.mean(
            self.free_energy(chain_end))
        # We must not compute the gradient through the gibbs sampling
        gparams = T.grad(cost, self.params, consider_constant=[chain_end])
        # end-snippet-3 start-snippet-4
        # constructs the update dictionary
        for gparam, param in zip(gparams, self.params):
            # make sure that the learning rate is of the right dtype
            updates[param] = param - gparam * T.cast(
                lr,
                dtype=theano.config.floatX
            )
        if persistent:
            # Note that this works only if persistent is a shared variable
            updates[persistent] = nh_samples[-1]
            # pseudo-likelihood is a better proxy for PCD
            monitoring_cost = self.get_pseudo_likelihood_cost(updates)
        else:
            # reconstruction cross-entropy is a better proxy for CD
            monitoring_cost = self.get_reconstruction_cost(updates,
                                                           pre_sigmoid_nvs[-1])

        return monitoring_cost, updates
        # end-snippet-4

    def get_pseudo_likelihood_cost(self, updates):
        """Stochastic approximation to the pseudo-likelihood"""

        # index of bit i in expression p(x_i | x_{\i})
        bit_i_idx = theano.shared(value=0, name='bit_i_idx')

        # binarize the input image by rounding to nearest integer
        xi = T.round(self.input)

        # calculate free energy for the given bit configuration
        fe_xi = self.free_energy(xi)

        # flip bit x_i of matrix xi and preserve all other bits x_{\i}
        # Equivalent to xi[:,bit_i_idx] = 1-xi[:, bit_i_idx], but assigns
        # the result to xi_flip, instead of working in place on xi.
        xi_flip = T.set_subtensor(xi[:, bit_i_idx], 1 - xi[:, bit_i_idx])

        # calculate free energy with bit flipped
        fe_xi_flip = self.free_energy(xi_flip)

        # equivalent to e^(-FE(x_i)) / (e^(-FE(x_i)) + e^(-FE(x_{\i})))
        cost = T.mean(self.n_visible * T.log(T.nnet.sigmoid(fe_xi_flip -
                                                            fe_xi)))

        # increment bit_i_idx % number as part of updates
        updates[bit_i_idx] = (bit_i_idx + 1) % self.n_visible

        return cost

    def get_reconstruction_cost(self, updates, pre_sigmoid_nv):
        """Approximation to the reconstruction error

        Note that this function requires the pre-sigmoid activation as
        input.  To understand why this is so you need to understand a
        bit about how Theano works. Whenever you compile a Theano
        function, the computational graph that you pass as input gets
        optimized for speed and stability.  This is done by changing
        several parts of the subgraphs with others.  One such
        optimization expresses terms of the form log(sigmoid(x)) in
        terms of softplus.  We need this optimization for the
        cross-entropy since sigmoid of numbers larger than 30. (or
        even less then that) turn to 1. and numbers smaller than
        -30. turn to 0 which in terms will force theano to compute
        log(0) and therefore we will get either -inf or NaN as
        cost. If the value is expressed in terms of softplus we do not
        get this undesirable behaviour. This optimization usually
        works fine, but here we have a special case. The sigmoid is
        applied inside the scan op, while the log is
        outside. Therefore Theano will only see log(scan(..)) instead
        of log(sigmoid(..)) and will not apply the wanted
        optimization. We can not go and replace the sigmoid in scan
        with something else also, because this only needs to be done
        on the last step. Therefore the easiest and more efficient way
        is to get also the pre-sigmoid activation as an output of
        scan, and apply both the log and sigmoid outside scan such
        that Theano can catch and optimize the expression.

        """

        cross_entropy = T.mean(
            T.sum(
                self.input * T.log(T.nnet.sigmoid(pre_sigmoid_nv)) +
                (1 - self.input) * T.log(1 - T.nnet.sigmoid(pre_sigmoid_nv)),
                axis=1
            )
        )

        return cross_entropy


# copied from $WORKSPACE/DeepLearningTutorials/code/logistic_sgd.py
def shared_dataset(data_xy, borrow=True):
    """ Function that loads the dataset into shared variables

    The reason we store our dataset in shared variables is to allow
    Theano to copy it into the GPU memory (when code is run on GPU).
    Since copying data into the GPU is slow, copying a minibatch everytime
    is needed (the default behaviour if the data is not in a shared
    variable) would lead to a large decrease in performance.
    """
    data_x, data_y = data_xy
    if type(data_x) in [list, tuple]:
        data_x = numpy.vstack(data_x)
    if type(data_y) in [list, tuple]:
        data_y = numpy.vstack(data_y).ravel()
    shared_x = theano.shared(numpy.asarray(
        data_x, dtype=theano.config.floatX), borrow=borrow)
    shared_y = theano.shared(numpy.asarray(
        data_y, dtype=theano.config.floatX),
                             borrow=borrow)
    # When storing data on the GPU it has to be stored as floats
    # therefore we will store the labels as ``floatX`` as well
    # (``shared_y`` does exactly that). But during our computations
    # we need them as ints (we use labels as index, and if they are
    # floats it doesn't make sense) therefore instead of returning
    # ``shared_y`` we will have to cast it to int. This little hack
    # lets ous get around this issue
    return(shared_x, T.cast(shared_y, 'int32'))


def load_data(n_samples, batch_size, dataset=None):
    try:

        os.environ['FUEL_DATA_PATH'] = os.sep.join([os.environ['WORKSPACE'],
                'fuel/fuel/datasets'])
        datasets = []
        for mnist in [MNIST("train"), MNIST("test")]:
            if mnist.which_set == 'train':
                # adding training
                train = tuple(zip(*Flatten(
                    DataStream.default_stream(mnist,
                        iteration_scheme=SequentialScheme(
                            3 * n_samples * batch_size,
                            batch_size=batch_size))
                        ).get_epoch_iterator()))
                datasets.append((train[0][:2 * n_samples], train[1][:2 *
                    n_samples]))
                datasets.append((train[0][2 * n_samples:], train[1][2 *
                    n_samples:]))
            elif mnist.which_set == 'test':
                datasets.append(tuple(zip(*Flatten(
                    DataStream.default_stream(mnist,
                        iteration_scheme=SequentialScheme(
                            n_samples * batch_size,
                            batch_size=batch_size))
                        ).get_epoch_iterator())))
        for i in range(len(datasets)):
            datasets[i] = shared_dataset(datasets[i])

    except ImportError:
        import logistic_sgd
        datasets = logistic_sgd.load_data(dataset)
    return(datasets)


def load_streams(n_samples, batch_size):
    """
    return fuel.streams.DataStream instead of shared datasets
    """
    os.environ['FUEL_DATA_PATH'] = os.sep.join([os.environ['WORKSPACE'],
            'fuel/fuel/datasets'])
    datastreams = []
    for mnist in [MNIST("train"), MNIST("test")]:
        if mnist.which_set == 'train':
            # adding training
            train = list(chain.from_iterable(Flatten(DataStream.default_stream(
                mnist, iteration_scheme=SequentialScheme(
                    3 * n_samples * batch_size,
                    batch_size=3 * n_samples * batch_size))
                                                ).get_epoch_iterator()))
            datastreams.append(DataStream(IterableDataset(OrderedDict([
                (name, samples[:2 * n_samples * batch_size]) for name, samples
                in zip(['features', 'targets'], train)])),
                iteration_scheme=SequentialScheme(2 * n_samples * batch_size,
                    batch_size=batch_size)))  # train
            datastreams.append(DataStream(IterableDataset(OrderedDict([
                (name, samples[2 * n_samples * batch_size:]) for name, samples
                in zip(['features', 'targets'], train)])),
                iteration_scheme=SequentialScheme(n_samples * batch_size,
                    batch_size=batch_size)))  # valid
        elif mnist.which_set == 'test':
            datastreams.append(Flatten(DataStream.default_stream(mnist,
                    iteration_scheme=SequentialScheme(
                        n_samples * batch_size,
                        batch_size=batch_size))))
    return(datastreams)


def test_rbm(learning_rate=0.1, training_epochs=15,
             dataset='mnist.pkl.gz', batch_size=20,
             n_chains=20, n_samples=10, output_folder='rbm_plots',
             n_hidden=500):
    """
    Demonstrate how to train and afterwards sample from it using Theano.

    This is demonstrated on MNIST.

    :param learning_rate: learning rate used for training the RBM

    :param training_epochs: number of epochs used for training

    :param dataset: path the the pickled dataset

    :param batch_size: size of a batch used to train the RBM

    :param n_chains: number of parallel Gibbs chains to be used for sampling

    :param n_samples: number of samples to plot for each chain

    """

    from symlearn.blocks import model
    from symlearn.blocks.main_loop import MainLoop
    from symlearn.blocks.algorithms import GradientDescent
    from symlearn.blocks.extensions import FinishAfter, Printing, Timing
    from symlearn.blocks.extensions.monitoring import TrainingDataMonitoring

    datastreams = load_streams(n_samples, batch_size)

    # allocate symbolic variables for the data
    x = T.matrix('x')  # the data is presented as rasterized images

    # initialize storage for the persistent chain (state = hidden
    # layer of chain)
    persistent_chain = theano.shared(numpy.zeros((batch_size, n_hidden),
                                                 dtype=theano.config.floatX),
                                     borrow=True)

    # construct the RBM class
    rbm = RBM(input=x, n_visible=28 * 28, n_hidden=n_hidden)

    # get the cost and the gradient corresponding to one step of CD-15
    cost, updates = rbm.get_cost_updates(lr=learning_rate,
                                         persistent=persistent_chain, k=15)

    #################################
    #     Training the RBM          #
    #################################
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)

    # start-snippet-5
    # it is ok for a theano function to have no output
    # the purpose of train_rbm is solely to update the RBM parameters
    rbm_model = model.Model(cost)
    sgd = GradientDescent(cost, rbm_model.get_parameter_values(),
            gradients=updates)
    extentions = [TrainingDataMonitoring(cost), PlotExtension(output_folder),
                Timing(), Printing(),
                FinishAfter(after_n_epochs=training_epochs)]
    main_loop = MainLoop(sgd, datastreams[0], rbm_model, extentions=extentions)

    start_time = time.clock()
    main_loop.run()
    end_time = time.clock()
    pretraining_time = (end_time - start_time) - extentions[1].plotting_time

    print(('Training took %f minutes' % (pretraining_time / 60.)))
    # end-snippet-5 start-snippet-6
    #################################
    #     Sampling from the RBM     #
    #################################
    # find out the number of test samples
    test_set_x = numpy.vstack(list(datastreams[-1].get_epoch_iterator()))
    number_of_test_samples = test_set_x.shape[0]

    # pick random test examples, with which to initialize the persistent chain
    test_idx = rng.randint(number_of_test_samples - n_chains)
    persistent_vis_chain = theano.shared(test_set_x[test_idx:test_idx +
        n_chains])
    # end-snippet-6 start-snippet-7
    plot_every = 1000
    # define one step of Gibbs sampling (mf = mean-field) define a
    # function that does `plot_every` steps before returning the
    # sample for plotting
    (
        [
            presig_hids,
            hid_mfs,
            hid_samples,
            presig_vis,
            vis_mfs,
            vis_samples
        ],
        updates
    ) = theano.scan(
        rbm.gibbs_vhv,
        outputs_info=[None, None, None, None, None, persistent_vis_chain],
        n_steps=plot_every
    )

    # add to updates the shared variable that takes care of our persistent
    # chain :.
    updates.update({persistent_vis_chain: vis_samples[-1]})
    # construct the function that implements our persistent chain.
    # we generate the "mean field" activations for plotting and the actual
    # samples for reinitializing the state of our persistent chain
    sample_fn = theano.function(
        [],
        [
            vis_mfs[-1],
            vis_samples[-1]
        ],
        updates=updates,
        name='sample_fn'
    )
    PlotImages(number_of_test_samples, n_chains, sample_fn, output_folder)

if __name__ == '__main__':
    test_rbm()
