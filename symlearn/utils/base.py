from collections import Counter, UserDict
from contextlib import contextmanager
from functools import partial

import numpy
import pandas

import re
import os
import logging
import shelve
import csv

__path__= ['.', '../scripts/run_shelter_comp']

logger = logging.Logger(__name__)
data_dir = os.path.join(os.getenv('DATADIR', default='.'), 'data')

NUM_PAT = re.compile(r"([()+\-.,]?\d+)+")
PUNKT_PAT = re.compile(r"^([^\w]+)$")


class VocabularyDict(UserDict):

    def __init__(self, dict_file, counter=None, max_features=numpy.Inf,
            norm_number=None, norm_punkt=None):
        self.max_features = max_features 
        if type(dict_file) is str:  # open dictionary file via shelve
            if counter:
                func = partial(preprocess_dictionary, counter, dict_file)
            else:
                func = partial(shelve.open, dict_file, flag='r')
            with func() as shelf:
                super(VocabularyDict, self).__init__(shelf)
        elif isinstance(dict_file, (VocabularyDict, dict)): # copy constructor
            super(VocabularyDict, self).__init__(dict_file)

        if not '-unk-' in self.data:
            self.data['-unk-'] = -1  # reference to the last column in embedding
            
        if norm_number:
            self.norm_number = NUM_PAT
        else: 
            self.norm_number = None
        if norm_punkt:
            self.norm_punkt = PUNKT_PAT
        else:
            self.norm_punkt = None 

    def __missing__(self, word):
        if self.norm_number and self.norm_number.match(word):
            return(self.data['-num-'])
        if self.norm_punkt and (self.norm_punkt.match(word) or word in
                ['-lrb-', '-rrb-']):
            return(self.data['-punkt-'])
        return self.data['-unk-']

    def __setitem__(self, word, val):
        if word == '-unk-':
            val = -1
        if val < self.max_features:
            super(VocabularyDict, self).__setitem__(word, val)

    def __getitem__(self, word):
        val = super(VocabularyDict, self).__getitem__(word)
        if val == -1:
            return int(numpy.min([len(self) - 1, self.max_features]))
        else:
            return int(val)


@contextmanager
def preprocess_dictionary(counter, dictionary_file, max_features=None):
    assert(isinstance(counter, Counter))
    with shelve.open(dictionary_file, flag='n') as shelf:
        for i, (w, _) in enumerate(counter.most_common(n=max_features)):
            shelf[w] = i
        if not max_features:
            max_features = len(counter)
        logger.info("creating dictionary file %s" % dictionary_file)
        yield shelf


def compute_inverse(vocab):
    index2vocab = numpy.empty(len(vocab), dtype=numpy.object)
    index2vocab[tuple(vocab.values()), ] = list(vocab.keys())
    return(index2vocab)


def run_batch(handler, *args, **kwargs):
    from csgraph.adjmatrix import to_csgraph
    import recursnn.cnltk_trees as cnltk
    filename = kwargs.pop('filename', os.path.join(data_dir, 'trees.bin'))
    batch_size = kwargs.get('batch_size', 1000)
    ttl_trees = kwargs.pop('ttl_trees', 11855)  # TODO: read from file
    n_batch = numpy.ceil(ttl_trees / batch_size).astype(numpy.int)
    with shelve.open(os.path.join(data_dir, 'vocab'), flag='n') as vocab:
        for i in range(n_batch):
            logger.info('%d th batch round are started' % (i))
            data = cnltk.read_bin_trees(filename,
                    list(range(i * batch_size, (i + 1) * batch_size)),
                    converted=False)
            data = [to_csgraph(d, vocab) for d in data]
            logger.info('%d th batch of trees are generated' % (i))
            index2vocab = compute_inverse(vocab)
            handler_args = args + (index2vocab,)
            handler(i, data, *handler_args, **kwargs)
            logger.info('%d th batch of stat_result are stored' % (i))


@contextmanager
def check_treebased_phrases(csv_file, n_rows=None, memory_map=False):
    """
    ensure the phrases are all included if the sentence id is also included in
    data set

    Parameters
    ==========
    @param csv_file: string type
        used to pass to pandas read_table
    @param n_rows: int or None 
        used to indicate how many rows should be read in. If None, all the
        records will be read in
    @param memory_map: boolean
        used to indicate if usin memory map with pandas.read_csv  
    """
    cur_dialect = csv.unix_dialect()
    cur_dialect.skipinitialspace=True
    dtypes_ = {'tree_id': numpy.int16, 'node_id': numpy.int16, 'sentiment': numpy.int8,
            'level': numpy.int16, 'start_pos': numpy.int32, 'end_pos': numpy.int32}
    phrases = pandas.read_table(csv_file, delimiter=',', header=0, dialect=cur_dialect,
            nrows=n_rows, memory_map=memory_map, dtype= dtypes_,
            index_col=False)  # adding index_col=False to disable using first
                              # column as index
    treeid_grps =  phrases.groupby(phrases['tree_id'].values)
    ids, tree_sizes = map(numpy.asarray, zip(*[(gid, len(grp) - 1) for gid, grp in
       treeid_grps]))
    
    if n_rows: # only check if n_rows is not None
        # 1. checking the root has node_id equals 0
        sentences = treeid_grps.nth(0)
        assert(all(sentences['node_id'] == 0))
        
        # 2. excluding those truncating records due to not all of n_rows read in
        test_phrases = phrases[phrases['node_id'] != 0]

        ids = ids[tree_sizes!=0]
        phrases = pandas.concat([treeid_grps.get_group(i) for i in ids])
        assert(len(test_phrases)== numpy.sum(tree_sizes))
        tree_sizes = tree_sizes[tree_sizes != 0]

        # 3. checking the start_pos and end_pos will map to the same extracted phrases
        sentences_toks = sentences['phrase'].apply(lambda x: x.split()) 
        assert(all(test_phrases.apply(
            lambda rec, toks: ' '.join(
                toks[rec['tree_id']][rec['start_pos']:rec['end_pos']]) == rec['phrase'],
            axis=1, broadcast=False, raw=False, reduce=False,
            args=(sentences_toks,))))

    yield phrases, tree_sizes


def tile_raster_images(X, img_shape, tile_shape, tile_spacing=(0, 0),
                       scale_rows_to_unit_interval=True,
                       output_pixel_vals=True):
    """
    copied from deeplearning tutorial
    Transform an array with one flattened image per row, into an array in
    which images are reshaped and layed out like tiles on a floor.

    This function is useful for visualizing datasets whose rows are images,
    and also columns of matrices for transforming those rows
    (such as the first layer of a neural net).

    :type X: a 2-D ndarray or a tuple of 4 channels, elements of which can
    be 2-D ndarrays or None;
    :param X: a 2-D array in which every row is a flattened image.

    :type img_shape: tuple; (height, width)
    :param img_shape: the original shape of each image

    :type tile_shape: tuple; (rows, cols)
    :param tile_shape: the number of images to tile (rows, cols)

    :param output_pixel_vals: if output should be pixel values (i.e. int8
    values) or floats

    :param scale_rows_to_unit_interval: if the values need to be scaled before
    being plotted to [0,1] or not


    :returns: array suitable for viewing as an image.
    (See:`Image.fromarray`.)
    :rtype: a 2-d array with same dtype as X.

    """

    assert len(img_shape) == 2
    assert len(tile_shape) == 2
    assert len(tile_spacing) == 2

    # The expression below can be re-written in a more C style as
    # follows :
    #
    # out_shape    = [0,0]
    # out_shape[0] = (img_shape[0]+tile_spacing[0])*tile_shape[0] -
    #                tile_spacing[0]
    # out_shape[1] = (img_shape[1]+tile_spacing[1])*tile_shape[1] -
    #                tile_spacing[1]
    out_shape = [
        (ishp + tsp) * tshp - tsp
        for ishp, tshp, tsp in zip(img_shape, tile_shape, tile_spacing)
    ]

    if isinstance(X, tuple):
        assert len(X) == 4
        # Create an output numpy ndarray to store the image
        if output_pixel_vals:
            out_array = numpy.zeros((out_shape[0], out_shape[1], 4),
                                    dtype='uint8')
        else:
            out_array = numpy.zeros((out_shape[0], out_shape[1], 4),
                                    dtype=X.dtype)

        #colors default to 0, alpha defaults to 1 (opaque)
        if output_pixel_vals:
            channel_defaults = [0, 0, 0, 255]
        else:
            channel_defaults = [0., 0., 0., 1.]

        for i in range(4):
            if X[i] is None:
                # if channel is None, fill it with zeros of the correct
                # dtype
                dt = out_array.dtype
                if output_pixel_vals:
                    dt = 'uint8'
                out_array[:, :, i] = numpy.zeros(
                    out_shape,
                    dtype=dt
                ) + channel_defaults[i]
            else:
                # use a recurrent call to compute the channel and store it
                # in the output
                out_array[:, :, i] = tile_raster_images(
                    X[i], img_shape, tile_shape, tile_spacing,
                    scale_rows_to_unit_interval, output_pixel_vals)
        return out_array

    else:
        # if we are dealing with only one channel
        H, W = img_shape
        Hs, Ws = tile_spacing

        # generate a matrix to store the output
        dt = X.dtype
        if output_pixel_vals:
            dt = 'uint8'
        out_array = numpy.zeros(out_shape, dtype=dt)

        for tile_row in range(tile_shape[0]):
            for tile_col in range(tile_shape[1]):
                if tile_row * tile_shape[1] + tile_col < X.shape[0]:
                    this_x = X[tile_row * tile_shape[1] + tile_col]
                    if scale_rows_to_unit_interval:
                        # if we should scale values to be between 0 and 1
                        # do this by calling the `scale_to_unit_interval`
                        # function
                        this_img = scale_to_unit_interval(
                            this_x.reshape(img_shape))
                    else:
                        this_img = this_x.reshape(img_shape)
                    # add the slice to the corresponding position in the
                    # output array
                    c = 1
                    if output_pixel_vals:
                        c = 255
                    out_array[
                        tile_row * (H + Hs): tile_row * (H + Hs) + H,
                        tile_col * (W + Ws): tile_col * (W + Ws) + W
                    ] = this_img * c
        return out_array


def compute_len_stats(len_dist):
    """
    return lengths distribution of sentence plus cumulative sum
    @param len_dist is a list whose elements are corresponding sentence length
    """
    # TODO: consider to replace with numpy.bincount
    len_count = numpy.zeros(len(numpy.unique(len_dist)) + 1, dtype=numpy.int)
    for i, cur_len in enumerate(numpy.unique(len_dist)):
        len_count[i + 1] = numpy.sum(len_dist == cur_len)
    return(numpy.unique(len_dist), len_count)
