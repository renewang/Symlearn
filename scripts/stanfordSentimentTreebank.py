from collections import defaultdict, Counter
from itertools import chain 

from nltk.parse.stanford import StanfordParser
from nltk.tree import Tree
from nltk.corpus import stopwords

import nltk.parse.stanford as stf 

from symlearn.utils import (check_treebased_phrases, VocabularyDict, WordNormalizer)

import numpy
import pandas

import os

version ='3.6.0'
data_dir = os.path.join(os.getenv('DATADIR', default='..'), 'data')
parser_path = os.path.join(os.getenv('HOME'),
        '.m2','repository','edu','stanford','nlp','stanford-corenlp', version)


def run_pcfg_grammars(csvfile, **kwargs):
    # not really useful tagging
    # train_corpus = nltk.corpus.treebank.tagged_sents(tagset='universal')
    # ct.train(train_corpus[:1000],'data/model.crf.tagger')
    # ct.tag_sents([word_tokenize(phrases[0])])
    from recursnn.cnltk_trees import read_bin_trees
    n_rows = kwargs.pop('n_rows', None)
    is_verbose = kwargs.pop('verbose', False)
    with preprocess_data(os.path.join(data_dir, csvfile), n_rows=n_rows)  as handler:
        ids, phrases, sentiments, levels, weights = handler

    parser = StanfordParser(path_to_jar=os.path.join(parser_path, 'stanford-parser.jar'),
                            path_to_models_jar=os.path.join(parser_path,'stanford-parser-3.5.2-models.jar'),
                            corenlp_options='-printPCFGkBest 1')
    btrees = []
    for i, iter_ in enumerate(parser.raw_parse_sents(phrases[levels==0],
        verbose=is_verbose)):
        t = list(iter_)
        assert(t[0].label() == 'ROOT')
        btrees.append(t[0][0])
        for path in btrees[-1].treepositions():
            assert(not isinstance(btrees[-1][path], Tree) or len(btrees[-1][path]) <= 2)

    parsing_trees = [Tree.fromstring(t) for t in read_bin_trees(
            os.path.join(data_dir, 'trees.bin'), numpy.unique(ids),
            converted=False)]

    pos_pair = defaultdict(int)
    for pt, bt in zip(parsing_trees, btrees):
        assert(pt.treepositions() == bt.treepositions())
        for path in pt.treepositions():
            pos_pair[(len(path), pt[path].label(), bt[path].label())] +=1
    return(btrees, parsing_trees)


def create_vocab_variants(csv_file, n_rows=None):
    """
    create four different vocabulary (full set, normalize number only,
    normalize punctuation only, normalize both number and punctuation)
    """
    
    with check_treebased_phrases(csv_file, n_rows=n_rows) as handler:
        phrase_df, phrase_sizes = handler 
        # handle phrases to return word index and store the vocabulary on the
        # disk
        params = [(None, {'norm_number': False, 'norm_punkt': False}),
                  ('-num-', {'norm_number': True, 'norm_punkt': False}),
                  ('-punkt-', {'norm_number': False, 'norm_punkt': True}),
                  ('-num-punkt-',{'norm_number': True, 'norm_punkt': True}),
                  ('-no-stopwords-', {'norm_number': True, 'norm_punkt': True, 
                    'stopwords': stopwords.words('english')})]
        for suffix, kwparams in params:
            stopwords_ = kwparams.pop('stopwords', [])
            analyzer = WordNormalizer(tokenizer=lambda x: x.strip().lower().split(),
                    **kwparams)
            phrases = phrase_df['phrase'].apply(lambda s: [w for w in analyzer(s)
                if not w in stopwords_])
            c = Counter(chain.from_iterable(phrases))
            basename, _ = os.path.splitext(csv_file)
            fn = "%s_vocab" %(basename)
            if suffix:
                fn = "%s%s" %(fn, suffix)
            yield VocabularyDict(fn, counter=c, **kwparams)


def preprocess_data(csv_file, **kwargs):
    """
    read in output result in csv format from C++ code

    @param csv_file: is filepath to output tree parsing result
    @param kwargs: keyword arguments only n_rows takes effect (for now)
    """
    n_rows = kwargs.get('n_rows', None)   # how many lines read in
    cutoff = kwargs.get('cutoff', None)  # merge levels
    max_features = kwargs.get('max_features', None) # max # of vocabulary  
    norm_number = kwargs.get('norm_number', False)
    norm_punkt = kwargs.get('norm_punkt', False)
    if n_rows < 0:
        n_rows = None

    with check_treebased_phrases(csv_file, n_rows=n_rows) as handler:
        phrase_df, phrase_sizes = handler 
        # 1. compute weight_by_size, each phrase is equally weighted by the
        # totoal size of trees 
        weight_by_size = numpy.ones(len(phrase_df))
        weight_by_size[phrase_df['node_id']!=0] = numpy.repeat(1/
                phrase_sizes, phrase_sizes) 
        
        # 2. compute weight_by_node, each phrase is weighted by the size of
        # word spans under the rooted subtree
        weight_by_node = numpy.ones(len(phrase_df))
        cur_size = 0
        for gid, subgrp in phrase_df.groupby('tree_id'):
            word_size = (subgrp['end_pos'] -
                    subgrp['start_pos']).values.astype(numpy.float)
            word_size /= word_size[0]
            word_size[1:] /= word_size[1:].sum()
            weight_by_node[cur_size:cur_size + len(subgrp)] = word_size
            cur_size+= len(subgrp)

        assert(numpy.allclose(numpy.sum(weight_by_size), 2 *
            (phrase_df['node_id']==0).sum()))
        assert(numpy.allclose(numpy.sum(weight_by_node), 2 *
            (phrase_df['node_id']==0).sum()))

        weights = {'weight_by_size': weight_by_size,
                   'weight_by_node': weight_by_node}
        ids = phrase_df['tree_id'].values
        sentiments = phrase_df['sentiment'].values
        levels = phrase_df['level'].values
        if cutoff:
            levels[levels > cutoff] = cutoff

        # still need the original sentence copy in string 
        sentences = phrase_df['phrase'][levels==0].apply(lambda x:
                x.lower().strip().split()).values
        phrases_pos = {}
        for groupid, subgroup in phrase_df.groupby(['tree_id']):
            phrases_pos[groupid] = list(zip(*(subgroup['start_pos'],
                subgroup['end_pos'])))
        return(ids, sentences, sentiments, levels, weights, phrases_pos)


class StanfordPCFGParser(stf.GenericStanfordParser):
    """
    a overriding class to provide flexible interface for parsing
    """
<<<<<<< HEAD
    treebank_dir = os.path.join(os.getenv('WORKSPACE'),
=======
    treebank_dir = os.path.join(os.getenv('WORKSPACE', default='..'),
>>>>>>> adding missing files
        'Kaggle','stanford_parser','stanfordSentimentTreebank')
    def __init__(self, **kwargs):
        super(__class__, self).__init__(**kwargs)


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
