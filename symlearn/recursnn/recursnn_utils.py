from collections import defaultdict
from collections import OrderedDict
from collections import namedtuple
from collections import deque
from itertools import permutations
from itertools import repeat
from itertools import chain
from functools import singledispatch
from operator import itemgetter, attrgetter
from heapq import heappush, heappop
from nltk.tree import Tree
from nltk.tree import ParentedTree
from scipy.sparse.csgraph import breadth_first_order
from contextlib import contextmanager

from sklearn.base import TransformerMixin
from sklearn.preprocessing import normalize

from . import get_phrases_helper
from . import _hash_phrase
import symlearn.csgraph.adjmatrix as pyadj


import scipy as sp
import numpy as np
import shelve
import logging
import mmap
import os
import sys

logger = logging.getLogger(__name__)
data_dir = os.path.join(os.getenv('WORKSPACE'), 'Kaggle/symlearn/data/')

try:
    import cnltk
    logger.info("%s using extension TreeOp"%(__name__))
except ImportError as e:
    logger.info("%s using python TreeOp"%(__name__))


class LastUpdatedOrderedDict(OrderedDict):
    """
    Store items in the order the keys were last added
    Recipes suggested from Python STD Lib example:
        ref:
        http://docs.python.org/3/library/collections.html?collections.OrderedDict
    """
    def __setitem__(tree, key, value):
        if key in tree:
            del tree[key]
        OrderedDict.__setitem__(tree, key, value)


def read_bin_file(filename, select_func=None):
    """
    read file as mmap file
    """

    def select_all(line=None, *args, **kwargs):
        return(True)

    if select_func is None:
        select_func = select_all

    with open(filename, 'r+b') as f:
        mm = mmap.mmap(f.fileno(), 0)
        pos = 0
        line_no = 0
        mm.read()  # read in all files, try to figure out how large the file is
        size = mm.tell()
        mm.seek(pos)  # move back to the beginning
        while pos < size:
            line = mm.readline().decode(encoding='utf-8')
            sig = select_func(line)
            if sig < 0:
                break
            pos = mm.tell()  # read in one tree. report current position
            # mm.seek(pos)  # redundnat
            line_no += 1  # adding corresponding index


@get_phrases_helper.register(Tree)
def get_phrases_from_tree(first, trees, vocab=None):
    trees = [treeindex_helper(t) for t in trees]
    phrase_to_node = {}
    # collect phrases_keys
    for tree_idx, tree in enumerate(trees):
        for p in tree.treepositions():
            if isinstance(tree[p], Tree):
                phrase_key = _hash_phrase(' '.join(tree[p].leaves()))
                phrase_to_node.setdefault(phrase_key, [])
                phrase_to_node[phrase_key].append((tree_idx, p))
    return(phrase_to_node)


def relabel_sentiments(trees, num_labels, recompute=True, **kwargs):
    """
    in-place relabel the sentiments based on the positivity probability and
    given cutoffs (from dictionary.txt)

    the program will do the following:
    1. traverse trees based on the node (tree_node index) and get the phrase
    2. checking the phrase with the dictionary.txt and acquire the phrase id
    3. checking the acquired phrase id with the sentiment_labels.txt and
       acquire the positivity probability
    4. relabel the node based on the acquired positivity probability and
       cutoffs

    Parameters
    ----------
    @param trees is a list of nltk trees
    @param num_labels is integer for the total number of labels
    @return the modified nltk trees with new labels
    """
    if recompute:
        phrase_to_node = get_phrases_helper(
                trees[0], trees, None)
    else:
        phrase_to_node = kwargs['phrase_to_node']

    # lookup
    phrases_ = list(phrase_to_node.keys())
    # deprecated due to raw_dictionary is no longer use
    with shelve.open(os.path.join(data_dir, kwargs.get(
         'phrase_dict', 'raw_dictionary')), flag='r') as phrase_dict:
        phrase_ids, sentiments_, _ = zip(*[phrase_dict.get(
            phrase, (-1, -1, phrase)) for phrase in phrases_])
        for phrase, pos_prob in zip(phrases_, sentiments_):
            for tree_idx, path_key in phrase_to_node[phrase]:
                trees[tree_idx][path_key]._label = get_sentiment_label(
                            float(pos_prob), num_labels)
    return(phrase_to_node)


def preprocess_tree(trees, preprocessor, **kwargs):
    """
    preprocess tree leaves which are presented as words based on the
    preprocessor of scikit-learn CountVectorizer.build_preporcessor or any user
    specified callable which need to be specified before learning

    @params trees is the collection trees
    @params preprocessor is a callable which will preprocess tree leaves
    """

    wordlist = [np.asarray(t.leaves()) for t in trees]
    mapping = {w: preprocessor(w) for wd in wordlist for w in wd}
    vocab = OrderedDict([(word, idx) for idx, word in
        enumerate(set(mapping.values()))])

    processed_trees = []
    for tree in trees:
        tree_copy = tree.copy(deep=True)
        for i, p in enumerate(tree_copy.treepositions('leaves')):
            # for each leaf whose parent could only have one child
            word = tree[p]
            tree_copy[p] = vocab[mapping[word]]

        if kwargs.get('cast_label') is True:
            for s in tree_copy.subtrees(lambda t: t.height() >= 2):
                s._label = int(s._label)

        if kwargs.get('indexing') is True:
            treeindex_helper(tree_copy)

        processed_trees.append(tree_copy)
    return(processed_trees, vocab)

def preprocess_docs(docs, analyzer, **kwargs):
    """
    preprocess documents to tokenize, collect vocabulary, and assign index to
    the doc based on the tokens it has

    @params docs is the collection docs 
    @params analyzer is a callable which will preprocess and tokenize each doc 

    >>> word2index, vocab = preprocess_docs(['this is a test .',
    ... 'this is another test .'], lambda x: x.lower().split())
    >>> len(word2index) == 2
    True
    >>> len(vocab) == 6 
    True
    >>> index2word = dict([(v, k) for k,  v in vocab.items()])
    >>> 'this is a test .' == ' '.join([index2word[index] for index in word2index[0]])
    True
    >>> 'this is another test .' == ' '.join([index2word[index] for index in word2index[1]])
    ... 
    True
    """
    proc_docs = list(map(analyzer, [doc.decode(encoding='utf-8')
        if type(doc) == bytes else doc for doc in docs]))
    vocab = dict([(w, i) for i, w in enumerate(
                 set(chain.from_iterable(proc_docs)))])
    processed_docs = list(map(lambda doc: list(map(lambda w: vocab[w], doc)),
        proc_docs))
    return(processed_docs, vocab)

def convert_to_span(tree):
    """
    Given a nltk tree return the leaves span for each non-terminal node.
    In the returned mapping, key are the path for non-terminal nodes and
    leaf are indexed by their sentence order

    @param tree is a nltk.tree.Tree
    """

    leaftoindex = dict(
        [(leaf_path[: -1], i) for i, leaf_path in
         enumerate(tree.treepositions(order='leaves'))])
    spans = defaultdict(tuple)
    # counting each non-terminals and the find the span
    for p in tree.treepositions(order='postorder'):
        if isinstance(tree[p], Tree):
            children = list(range(len(tree[p])))
            span = []
            for i in children:
                toleaf = list(p)[:]
                while len(toleaf) < tree.height() - \
                        1 and not tuple(toleaf) in leaftoindex:
                    toleaf.append(i)
                span.append(leaftoindex[tuple(toleaf)])
            assert(len(span) == len(children))
            spans[p] = tuple(span)  # store the pair of leaf index
    # root must span all the leaves
    assert(spans[()] == (0, len(leaftoindex) - 1))
    return(spans)


def embed_words(word_order, embedding, is_to_norm='l2'):
    """
    for the purpose to have complicated word embedding trasformer

    @param word_order is an array whose elements are indices of word to
                         vocabulary dict and used to retrieve word vector from
                         embedding
    @embedding is the embedding matrix storing word vector
    @is_to_norm is a string which indicate the normalization method
    """
    if isinstance(embedding, TransformerMixin):
        n_features = embedding.components_.shape[1]
        transform_func = embedding.transform
    elif isinstance(embedding, np.ndarray):
        n_features = embedding.shape[0]

        def default_transform(x):
            return(x.dot(embedding).astype(np.float))

        transform_func = default_transform
    else:
        raise NotImplementedError

    # reshape x into (n_words \times n_components)
    x = sp.sparse.csr_matrix((np.ones(
            (len(word_order),), dtype=np.int),
            (np.arange(len(word_order)), word_order)),
            shape=(len(word_order), n_features), dtype=np.int)
    xt = transform_func(x)
    if is_to_norm:
        xt = normalize(xt, norm=is_to_norm, axis=1, copy=False)
    return(xt)


def build_greedy_tree(word_order, **kwargs):
    """
    as described in Socher's thesis using greedy method to pick up the nodes
    with the minimal construction errors

    @param word_order is an 1D array which stores the indices of term matrix in
                        the order of word apperance in the sentence
    @param rae is the trained/fitted auto-encoder or decomposer
    @param embedding is the embedding matrix with dim equals to n_vocab_size
                        times n_components
    """
    build_method = {(True, True, False): build_with_predefined,
                    (False, False, False): build_with_predefined,
                    (False, False, True): build_with_greedy}

    build_func = build_method.get((kwargs.get('autoencoder') is None,
                                   kwargs.get('embedding') is None,
                                   kwargs.get('build_order') is None))
    if build_func is None:
        raise ValueError('cannot build tree')
    else:
        return(build_func(word_order, **kwargs))


def build_with_predefined(word_order, build_order, embedding=None,
                          autoencoder=None):
    """
    build tree with given building order and optional embedding
    @param word_order is a list which convert word into index to embedding
                         matrix
    @param build_order is a list which stores the order of nodes should be
                          merged
    @param embedding is a matrix whose row is the size of vocabulary and column
                        is the dimension of embedding / word vector space
    @param autoencoder is a autoencoder provides compositie transform to build
                          recursive tree
    """
    if type(build_order) is tuple and len(build_order) == 2:
        non_terms, tree_mat = build_order
        # map non_terms to order of terminals
        mat = np.ma.MaskedArray(
                tree_mat.toarray(),
                mask=np.eye(tree_mat.shape[0], dtype=np.bool))
        terms = OrderedDict(
                [(idx, i) for i, idx in
                    enumerate(visit_leaves_inorder(tree_mat))])

        rep_lookup, build_order = {}, []
        for node in non_terms:
            candidates = np.arange(
                    mat.shape[0], dtype=np.intp)[mat[node] == 1]  # min, max
            assert(len(candidates) == 2)
            result = [None, None]
            # Left subtree has been constructed, the left most word of the
            # subtree will be brought up to become representative;
            # While the right most word will be ordered in the build order
            if candidates[0] in rep_lookup:
                result[0] = rep_lookup[candidates[0]][0]
            # Right subtree has been constructed, the left most word of the
            # subtree will be brought up to become representative;
            # While the right most word will be ordered in the build order
            if candidates[1] in rep_lookup:
                result[1] = rep_lookup[candidates[1]][0]
            # Left subtree has not been constructed, using the left child node
            if result[0] is None:
                result[0] = candidates[0]
            # Right subtree has not been constructed, using the right child
            # node
            if result[1] is None:
                result[1] = candidates[1]
            rep_lookup[node] = result

        build_order = np.asarray(
                [terms[rep_lookup[node][1]] for node in non_terms])
        word_order = [terms[k] for k in terms.keys()]
        assert([b is not np.ma.masked for b in build_order])

    def find_predefined_word(stack, n_iter):
        best_ptr = np.where(np.asarray(stack) == build_order[n_iter])[0] - 1
        return(best_ptr)

    ttl_cost = None
    # TODO: calculate the cost afterwards
    if hasattr(autoencoder, 'score_samples'):
        pass

    tree = build_tree(word_order, find_predefined_word)
    return(tree, ttl_cost)


def build_with_greedy(word_order, embedding, autoencoder):
    """
    build greedy tree with given embedding matrix and auto-encoder
    @param word_order is a list which convert word into index to embedding
                         matrix
    @param embedding is a matrix whose row is the size of vocabulary and column
                        is the dimension of embedding / word vector space
    @param autoencoder is a autoencoder provides compositie transform to build
                          recursive tree
    """
    x = embed_words(word_order, embedding)
    n_components = getattr(embedding, 'n_components', None) or \
        getattr(embedding, 'shape')[1]

    if not hasattr(autoencoder, 'components_'):  # unfitted,
        try:
            autoencoder.fit(x, embedding=embedding)
        except:
            raise RuntimeError("No valid composite function supplied",
                               *sys.exc_info())

    ttl_cost = np.zeros(len(word_order) - 1)

    def find_min_cost(stack, n_iter):
        min_costs = np.zeros((len(stack) - 1,))
        for i in range(len(stack) - 1):
            # calculate reconstruction cost c
            min_costs[i] = autoencoder.score_samples(
                x[[stack[i], stack[i + 1]], :].reshape(
                    (1, 2 * n_components)))
            logger.debug("reconstruction cost = {} index = {}".format(
                    min_costs[i], stack))
        best_ptr = np.argmin(min_costs)
        ttl_cost[n_iter] = min_costs[best_ptr]
        # reset
        x[stack[best_ptr]] = autoencoder.transform(
            x[[stack[best_ptr], stack[best_ptr + 1]], :].reshape(
                (1, 2 * n_components)))
        x[stack[best_ptr + 1]] = 0
        return(best_ptr)

    tree = build_tree(word_order, find_min_cost)
    tree = treeindex_helper(tree)
    return(tree, ttl_cost[-1])


def build_tree(word_order, get_best_ptr, init_sentiment=2):
    """
    the main implementation for tree construction
    @param word_order is a list which convert word into index to embedding
                         matrix
    @get_best_ptr is a callable which will return the best_ptr
    @init_sentiment is initial sentiment label to be assigned to the newly
                    built node
    """
    merged_tree = LastUpdatedOrderedDict()
    stack = [i for i in range(len(word_order))]

    for n_iter in range(len(word_order) - 1):  # if all the nodes are merged
        cur_tree, best_ptr = None, get_best_ptr(stack, n_iter)
        assert(best_ptr in range(0, len(stack)))
        logger.debug(
            "merge pair {}".format((stack[best_ptr], stack[best_ptr + 1])))

        # maintain vocab order
        if stack[best_ptr] in merged_tree and \
                stack[best_ptr + 1] in merged_tree:
            left_child = merged_tree[stack[best_ptr]]
            right_child = merged_tree[stack[best_ptr + 1]]
        elif stack[best_ptr] in merged_tree:
            left_child = merged_tree[stack[best_ptr]]
            right_child = Tree(init_sentiment, [word_order[
                        stack[best_ptr + 1]]])
        elif stack[best_ptr + 1] in merged_tree:
            left_child = Tree(init_sentiment, [word_order[
                        stack[best_ptr]]])
            right_child = merged_tree[stack[best_ptr + 1]]
        else:
            left_child = Tree(
                init_sentiment, [word_order[stack[best_ptr]]])
            right_child = Tree(
                init_sentiment, [word_order[stack[best_ptr + 1]]])
        cur_tree = Tree(init_sentiment, [left_child, right_child])
        merged_tree[stack[best_ptr]] = cur_tree
        # remove one node and keep the other as a representative
        stack.remove(stack[best_ptr + 1])

    # ensure all nodes are merged into one
    assert(len(merged_tree) and len(stack) == 1)
    # make sure the last tree has the same number of words
    final_tree = merged_tree.popitem()[-1]
    assert(len(word_order) == len(final_tree.leaves()))
    if np.any([type(leaf) != int for leaf in final_tree.leaves()]):
        for p in final_tree.treepositions('leaves'):
            # ensure to convert python built-in int
            final_tree[p] = int(final_tree[p])
    return(final_tree)


def build_graph(word_order, get_best_ptr, init_sentiment=2):
    merged_graphs = LastUpdatedOrderedDict()
    stack = [i for i in range(len(word_order))]

    for n_iter in range(len(word_order) - 1):  # if all the nodes are merged
        stack.remove(stack[best_ptr + 1])

    for i in range(len(build_queue)):
        _, from_node, to_node = heapq.heappop(build_queue)
        if not from_node in node_tracking:
            node_tracking[from_node] = len(node_tracking)
        if not to_ndoe in node_tracking:
            node_tracking[to_node] = len(node_tracking)
        graph[from_node, to_node] = 1
    assert((graph.diagonal() > 1).sum() == len(word_order))
    return(merged_graphs[-1])

def get_true_label(gold_tree, eval_tree=None):
    """
    single tree version of getLabels
    @param gold_tree is the nltk.tree.Tree instance with ground truth
    @param eval_tree is the nltk.tree.Tree instance undertested
    """
    Ty = []
    gold_spans = convert_to_span(gold_tree)
    assert(len(gold_spans) == 2 * len(gold_tree.leaves()) - 1)

    if eval_tree:
        eval_spans = convert_to_span(eval_tree)
        assert(len(eval_spans) == 2 * len(eval_tree.leaves()) - 1)
        iterable = eval_spans.items()
    else:
        iterable = gold_spans.items()

    for nt, nt_range in iterable:
        if len(nt_range) == 2:
            gold_path = gold_tree.treeposition_spanning_leaves(
                nt_range[0], nt_range[1] + 1)
        elif len(nt_range) == 1:
            gold_path = gold_tree.leaf_treeposition(nt_range[0])[:-1]

        Ty.append((gold_tree[gold_path]._index, gold_tree[gold_path]._label))

    return([label if isinstance(label, int) else int(label) for _, label in
            sorted(Ty, key=itemgetter(0))])


def getLabels(goldtrees, greedytrees=None, word2index=None):
    """
    return the true sentiment class for the given word spans of all
    non-terminals in greedytrees by following the cases below:
       1. for the non-terminal in greedy tree whose word span has the same
       range in gold tree, just return the corresponding true label of the
       corresponding non-terminal node in gold tree
       2. for the non-terminal in greedy tree whose word span cannot be found
       in gold tree, just return the corresponding true label with the greater
       range containing it.
    @params goldtrees is a collection of nltk trees whose labels will be
    extracted
    @params an option of greedytrees whose labels are also extracted and
    further compared with goldtrees
    @param word2index is is a list whose element is word order
    """
    Ty = []
    if (isinstance(goldtrees, dict) or isinstance(
            greedytrees, dict)) and word2index:
        iterable = word2index.keys()
    else:
        assert(not greedytrees or len(greedytrees) == len(goldtrees))
        iterable = range(len(goldtrees))
    for sentid in iterable:
        gold_tree = goldtrees[sentid]
        if greedytrees:
            eval_tree = greedytrees[sentid]
            Ty.extend(get_true_label(gold_tree, eval_tree))
        else:
            Ty.extend(get_true_label(gold_tree))
    return(np.hstack(Ty))


def set_predict_label(eval_tree, eval_labels):
    """
    set labels for eval_tree according to the eval_labels and return the
    sentiment class for whole sentence
    @param eval_tree is a nltk tree which will be assigned predicted
                        sentiment labels
    @param eval_labels is a collection labels will be assigned to the
           evaluation nltk tree
    """
    root_label = -1

    for path in eval_tree.treepositions():
        if isinstance(eval_tree[path], Tree):
            eval_tree[path]._label = eval_labels[eval_tree[path]._index]

    root_label = eval_tree[()]._label
    assert(root_label != -1)
    return(root_label)


def setLabels(greedytrees, word2index, labels):
    """
    set sentimet label for a tree
    @param greedytrees is a collection of nltk trees
    @param word2index is is a list whose element is word order
    @param labels are the predicted labels going to be assigned
           to the nltk tree
    """
    predicted_label = []
    for sentid in word2index.keys():
        eval_tree = greedytrees[sentid]
        eval_labels = labels[sentid]
        if eval_tree:
            predicted_label.append(set_predict_label(eval_tree, eval_labels))
        else:
            predicted_label.append(eval_labels)
    return(np.array(predicted_label).ravel())


def get_nonterminals(tree):
    """
    get the non-terminal nodes in nltk tree and map non-terminal indexing to
    the input matrix using heap's method to indexing nodes which the 0th node
    indicates root and for any ith node whose left child
    is 2*i+1 and 2*i+2 for right child. While for children jth node, simply
    take floor of being divided by 2.
    @param tree is a nltk tree whose non-terminal nodes will be indexed and
                   collected
    """
    childrenids = []
    for pos in tree.treepositions(order='preorder'):  # top down
        if isinstance(tree[pos], Tree):
            if len(tree[pos]) == 2:
                assert(hasattr(tree[pos], '_index'))
                heappush(childrenids, (tree[pos]._index, pos))
    assert(len(childrenids) == len(tree.leaves()) - 1)
    min_heaps = [heappop(childrenids) for h in range(len(childrenids))]
    # ensure the root is at the top of the stack
    if tree._index == 0:  # root's index is the smallest
        return(list(reversed(min_heaps)))
    return(min_heaps)


def get_number_tree_configurations(num_of_leaves):
    """
    Given num_of_leaves n, there are n-1 total iterations to construct tree.
    For each iteration i, n-i nodes and n-i pairs available for selection due
    to the restrictions on the two children nodes which must be consecutive and
    starting from i = 1 but not from the root level is because the freedom
    for the root level is lost. Therefore the possible tree configurations
    should be \prod_{i=1}^{n-1} n-i
    @param num_of_leaves is the size of leaf

    >>> get_number_tree_configurations(1)
    0
    >>> get_number_tree_configurations(2)
    1
    >>> get_number_tree_configurations(3)
    2
    >>> get_number_tree_configurations(4)
    6
    """
    def get_cur_number(cur_num):
        if cur_num > 2:
            return(get_cur_number(cur_num - 1) * (cur_num - 1))
        elif cur_num == 2:
            return(1)
        else:
            return(0)
    return(get_cur_number(num_of_leaves))


def _rotate_to(tree, top, random_state=None):
    """
    decide which direction to rotate
    @param tree is a nltk tree which will be manipulated
    @param top is the node whose subtrees will be rotated
    @random_state is the seed used to decide which direciton to turn
    """

    # raise excpetion
    if len(tree[top].leaves()) < 3:
        raise AssertionError

    for i in [0, 1]:
        if len(tree[top][i]) == 1:
            return(i)

    # draw a random number
    if np.random.RandomState(random_state).random_sample(
            1) > 0.5:  # return right
        return(1)
    else:
        return(0)  # return left


def rotate(tree, top, random_state=None):
    """
    a not in-place rotate op implementation for nltk tree
    @param tree is a nltk tree
    @top is the topest node whose position will be rotated position will be
    @random_state is a random seed passed to reproduce rotaion result
    rotated position will be rotated position will be rotated
    steps:
    1. decide which direction to rotate:
        a. Identify the outgroup. The one as terminal node, turn the tree
        towards the outgroup direction.  For example, if the outgroup is at the
        right, then turn clock-wise (right); On the other hand, turn counter
        clock-wise (left)
        b. If there are no outgroup present, then flip the
        fair coin and choose a child node to rotate
        c. output the picked node (pivot)
    2. rotate
        a. unlink the the top node (top) from its parent
        b. unlink the child node (grand) of picked node (pivot) at picked side
        from picked child.
        c. link the picked node (pivot) to the parent of top
        node
        d. linek the parent node to the pivot node at picked side
        e. link the child node selected in b to the top node in order to
        replace the pivot node's position
    """
    # tree rotate op
    # TODO: study memoryview for the in-place implementations
    # convert tree to the ParentedTree for better data structure with pointer
    # to the parent
    if not isinstance(tree, ParentedTree):
        raise TypeError
    # backup tree
    tree_copy = tree.copy(deep=True)

    try:
        where = _rotate_to(tree, top, random_state)

        parent = tree[top]
        pivot = tree[top][1 - where]
        grand = tree[top][1 - where][where]

        # take care root case:
        if parent.parent() is not None:  # 2a + 2c
            parent[1 - where] = None
            tree[top].parent()[tree[top].parent_index()] = pivot
        else:
            tree = pivot

        pivot[where] = None
        parent[1 - where] = grand  # 2e
        pivot[where] = parent  # 2d

        assert(parent.parent() == pivot)
        assert(grand.parent() == parent)
        assert(pivot.parent() == tree[top].parent() or pivot.parent() is None)
    except:
        tree = tree_copy
        raise AssertionError
    return(tree)


def treeindex_helper(tree):
    """
    adding postorder indexing
    @param tree is an existing nltk tree
    """
    # setting _index
    queue = deque(maxlen=2 * len(tree.leaves()) - 1)
    queue.append(())
    while len(queue) > 0:
        cur_pos = queue.pop()
        if isinstance(tree[cur_pos], Tree):
            if len(cur_pos) > 0:
                # left child will end as 0, 1 for right child
                tree[cur_pos]._index = tree[cur_pos[:-1]]._index * 2 + \
                        cur_pos[-1] + 1
            else:
                tree[cur_pos]._index = len(cur_pos)
            if len(tree[cur_pos]) == 2:
                for i in range(len(tree[cur_pos])):
                    queue.append(cur_pos + (i,))
    return(tree)


@singledispatch
def matrix_to_tree(matrix, hook=None, init_sentiment=2):
    """
    turn np.array or sp.sparse.* into nltk.tree.Tree from the following
    procedure:
    1. testing if the index is convertible to bit string
    2. if not, then calling two_passes_traversal and then call
    build_greedy_tree
    3. if yes, then encode bit string from index and return the tree directly
    """
    if is_heap_ordered(matrix):
        path2index = dict([(encode_path(i, ()), i) for i in
            np.arange(matrix.shape[0])[matrix.diagonal() > 0]])

        assert(len(path2index) == (matrix.diagonal() > 0).sum())
        cur_tree = Tree(init_sentiment, [])
        split_trees = LastUpdatedOrderedDict()
        not_visited = deque(sorted(path2index.keys()))
        # the root should be at the first
        if not_visited[0] == ():
            pop_func = not_visited.popleft
        else:
            pop_func = not_visited.pop

        while len(not_visited) > 0:
            cur_parent = pop_func()
            if cur_parent in split_trees:
                cur_tree = split_trees[cur_parent]
            # for internal node, there are two children at most
            for i in [0, 1]:
                if (cur_parent + (i,)) in path2index:
                    cur_tree.append(Tree(2, []))
            assert(len(cur_tree) == 2 or len(cur_tree) == 0)
            # for leaf, there won't be any children
            if len(cur_tree) == 0:
                # must be leaf
                word_index = matrix.diagonal()[path2index[cur_parent]]
                assert(word_index > 1)
                # adding 2 when constructing tree_matrix
                cur_tree.append(word_index - 2)
            else:
                for i in range(len(cur_tree)):
                    split_trees[cur_parent + (i, )] = cur_tree[i]
            split_trees[cur_parent] = cur_tree
            cur_tree = None
        greedy_tree = split_trees[()]
    else:
        build_order = two_passes_traversal(matrix, hook)
        greedy_tree, _ = build_greedy_tree(
                matrix.diagonal()[[matrix.diagonal() > 1]] - 2,
                build_order=(build_order, matrix))
    return(greedy_tree)


def _check_trees(tree, specified_maxlen=-1):
    """
    ensure the tree is processed and return the proper max_len for constructing
    sp.sparse.matrix
    """
    try:
        getattr(tree, '_index')
    except AttributeError:
        raise AttributeError('tree needs to be indexed')

    assert(np.all([type(leaf) in [int, np.intp] for leaf in tree.leaves()]))
    max_len = np.max([tree[pos[:-1]]._index for pos in
        tree.treepositions('leaf')]) + 1
    return(np.max([max_len, specified_maxlen]))


def label_to_matrix(tree, max_len=-1, given_rootlabel=None):
    """
    get sentiment labels from  nodes in nltk.tree.Tree and return the label for
    each node
    """
    max_len = _check_trees(tree, max_len)
    label_mat = sp.sparse.lil_matrix((max_len, max_len), dtype=np.int)
    for pos in tree.treepositions(order='preorder'):  # top down
        if isinstance(tree[pos], Tree):
            # to avoid zero label, all labels are added one
            label_mat[tree[pos]._index, tree[pos]._index] = \
                    int(tree[pos]._label) + 1
    if given_rootlabel is not None:
        label_mat[tree._index, tree._index] = given_rootlabel
    # shrink the matrix
    dias = label_mat.diagonal()
    dias = dias[dias > 0] - 1
    return(dias.view(np.ma.MaskedArray))


def tree_to_matrix(tree, vocab=None, max_len=-1, branch_factor=2):
    """
    turn nltk.tree.Tree nested list format into adjacent matrix (numpy.array)
    or adjacent linked list (sp.sparse.*, default) according to the giving a
    word order whose elements are index to vocabulary dict construct a
    constrained adjacent matrix which can furhter be expressed
    as nltk.tree.Tree

    @param tree is a nltk.tree.Tree
    @max_len is deprecated and will find the maximal index in the leaf
    @branch_factor is the number of children nodes and should be set as 2
    """
    if vocab is not None:
        return(pyadj.to_csgraph(tree, vocab))

    max_len = _check_trees(tree, max_len)
    connect_mat = sp.sparse.lil_matrix((max_len, max_len), dtype=np.intp)
    index2key = dict([(tree[p]._index, p) for p in tree.treepositions() if
            hasattr(tree[p], '_index')])
    for index in index2key.keys():
        for i in [1, 2]:
            child_idx = 2 * index + i
            if child_idx in index2key:  # as internal node
                connect_mat[index, child_idx] = 1
                connect_mat[index, index] = 1
            else:  # as leaf
                connect_mat[index, index] = tree[index2key[index] + (0,)] + 2
    # shrink the matrix
    dias = connect_mat.diagonal()
    connect_mat = connect_mat[:, dias!= 0]
    connect_mat = connect_mat[dias!=0, :]
    return(connect_mat)

def remove_diagonal(matrix):
    assert(matrix.format == 'csr')
    dias = matrix.diagonal()
    mat_no_selfloop = matrix - sp.sparse.csr_matrix((dias,
            (np.arange(matrix.shape[0]), np.arange(matrix.shape[1]))),
            shape=matrix.shape)
    assert(np.all(mat_no_selfloop.diagonal() ==
        np.zeros(matrix.shape[0], dtype=matrix.dtype)))
    return(mat_no_selfloop)

def visit_inodes_inorder(matrix, root=0):
    """
    given a tree matrix and return non-terminal nodes in indexing order
    the root will be returned at the last elements disregarding any indexing
    scheme

    @param matrix is the tree matrix in scipy.sparse format whose index should
    be the tree index
    @root is the index of root
    """
    if isinstance(matrix, pyadj.CSGraphProxy):
        if 'cnltk' in globals():
            logger.debug('calling boost python extension')
            mat = matrix.tocsr()
            return(cnltk.visit_inodes_inorder(cnltk.csr_adjmatrix(mat.data,
                mat.indices, mat.indptr, mat.shape)))
        else:
            logger.debug('calling proxy method')
            assert(hasattr(matrix, 'visit_inodes_inorder'))
            if root is None: # using default
                return(getattr(matrix, 'visit_inodes_inorder')())
            else:
                return(getattr(matrix, 'visit_inodes_inorder')(root=root))

    logger.debug('calling scipy directly')
    if not sp.sparse.issparse(matrix):
        matrix = sp.sparse.csr_matrix(matrix)
    if matrix.format != 'csr':
        matrix = matrix.tocsr()

    # determine where is root
    if root is None:
        root = 0 * (matrix.T.getrow(0).sum() == 1) + \
               (matrix.shape[0] - 1) * (
                       matrix.T.getrow(matrix.shape[0] - 1).sum() == 1)

    dias = matrix.diagonal()  # backup diagonal
    mat_no_selfloop = remove_diagonal(matrix)    
    # breadth first order: i is travesal order from root and n is location of
    # node in matrix
    nodes = list(map(itemgetter(1), sorted([(n, i) for i, n in enumerate(
        sp.sparse.csgraph.breadth_first_order(mat_no_selfloop, root, return_predecessors=False))
        if dias[n] == 1], key=itemgetter(0))))
    depths = [d for n, d in enumerate(
        sp.sparse.csgraph.shortest_path(mat_no_selfloop)[root, :].astype(np.int))
        if dias[n] == 1]
    assert(len(nodes) == len(depths))
    index2nts = sorted([(i, n, -d) for i, (n, d) in enumerate(
        zip(nodes, depths))], key=itemgetter(2, 1))
    # the index of non-terminals
    index2nts = np.asarray(index2nts)[:, 1]
    return(index2nts)


def visit_leaves_inorder(matrix, root=0):
    """
    recover leaves and re-arange them in the order within the sentence
    """
    if not sp.sparse.issparse(matrix):
        matrix = sp.sparse.csr_matrix(matrix)
    if matrix.format != 'csr':
        matrix = matrix.tocsr()

    # determine where is root
    if root is None:
        root = 0 * (matrix.T.getrow(0).sum() == 1) + \
               (matrix.shape[0] - 1) * (
                       matrix.T.getrow(matrix.shape[0] - 1).sum() == 1)

    dias = matrix.diagonal()  # backup diagonal
    mat_no_selfloop = remove_diagonal(matrix)    
    index_of_leaves = [n for i, n in enumerate(
        sp.sparse.csgraph.depth_first_order(matrix, root, return_predecessors=False))
        if dias[n] > 1]
    return(index_of_leaves)


def get_spanrange(matrix):
    """
    calculate the word span range for each node. 
    """

    if not sp.sparse.issparse(matrix):
        matrix = sp.sparse.csr_matrix(matrix)
    if matrix.format != 'csr':
        matrix = matrix.tocsr()

    all_leaves = np.zeros(matrix.shape[0], dtype=np.bool_)
    all_leaves[np.logical_or(matrix.diagonal() == 0,
               matrix.diagonal() > 1)] = True

    # retreive all non-terminals
    non_terms = visit_inodes_inorder(matrix)
    # locate leaves
    leaf_info = visit_leaves_inorder(matrix)

    map_leaf = {graph_idx: idx for idx, graph_idx in
            enumerate(leaf_info)}

    span_range = np.tile([len(map_leaf), 0], (len(non_terms), 1))

    for i in range(len(non_terms)):  # travel from bottom
        if not all_leaves[non_terms[i]]:  # checking if already updated
            children = [idx for idx in matrix.getrow(non_terms[i]).indices
                    if idx != non_terms[i]]
            # 0 is left, 1 is right
            span_range[i][0] = np.min([map_leaf.get(children[0], np.inf),
                span_range[i][0]])
            span_range[i][1] = np.max([map_leaf.get(children[1], 0),
                span_range[i][1]])

            # also update its ancenstors
            parent = non_terms[i]
            while parent != 0:
                parent = [idx for idx in
                        matrix.T.getrow(int(parent)).indices if idx !=
                        parent][0]  # exclude self
                assert(parent != non_terms[i])
                # update min
                if span_range[i, 0] < span_range[non_terms == parent, 0]:
                    span_range[non_terms == parent, 0] = span_range[i, 0]
                # update max
                if span_range[i, 1] > span_range[non_terms == parent, 1]:
                    span_range[non_terms == parent, 1] = span_range[i, 1]

            # is reaching the leaf or all children are updated
            all_leaves[non_terms[i]] = np.all(all_leaves[children])

    all_spans = np.ma.masked_all((len(non_terms) + len(leaf_info), 2),
                                    dtype=np.int)
    all_spans[non_terms] = span_range
    all_spans[tuple(map_leaf.keys()), ] = np.asarray(
            list(map_leaf.values()))[:, np.newaxis]
    all_spans += np.tile([0, 1], (len(all_spans), 1))
    all_spans[tuple(map_leaf.keys()), ] = np.ma.masked
    return all_spans 

def calculate_spanweight(matrix, all_spans=None):
    """
    calculate the word span for each node. For terminal node the span is zero

    """
    if isinstance(matrix, pyadj.CSGraphProxy):
        if 'cnltk' in globals():
            logger.debug('calling boost python extension')
            mat = matrix.tocsr()
            return(cnltk.calculate_spanweight(cnltk.csr_adjmatrix(mat.data,
                mat.indices, mat.indptr, mat.shape)))
        else:
            logger.debug('calling proxy method')
            return(getattr(matrix, 'calculate_spanweight')())

    logger.debug('calling scipy directly')

    if all_spans is None:
        all_spans = get_spanrange(matrix)

    span_weight = np.ones(len(all_spans), dtype=np.int)
    span_mask = span_weight.view(np.ma.MaskedArray)
    span_mask.mask = all_spans.mask.all(axis=1)
    nt_inds = np.arange(len(span_weight))[~span_mask.mask]
    for i, span_range in enumerate(all_spans.compressed().reshape(-1, 2)):
        # avoiding zero
        span_weight[nt_inds[i]] = span_range[1] - span_range[0]
    return(span_weight)

def retrieve_mapping(matrix):
    """
    return the parent-children mapping
    """

    nonterms = visit_inodes_inorder(matrix)
    if isinstance(matrix, pyadj.CSGraphProxy):
        matrix = matrix.tocsr()

    if not sp.sparse.issparse(matrix):
        matrix = sp.sparse.csr_matrix(matrix)
    if matrix.format != 'csr':
        matrix = matrix.tocsr()
    nodes_mask = np.ma.masked_where(matrix.diagonal() == 0,
            np.arange(matrix.shape[0]))

    mapping = np.tile(np.arange(matrix.shape[0])[:, np.newaxis], (1, 2))
    for inode in nonterms:
        mapping[inode] = np.asarray(
                [idx for idx in matrix.getrow(inode).indices if idx !=
                    inode])
    return(mapping[nodes_mask.compressed()])


def get_wordindices(matrix, vocab_size, span_range=None):
    """
    return word indices span by each node based on the tree matrix format
    """
    if span_range is None:
        span_range = get_spanrange(matrix)
    if hasattr(span_range, 'mask'):
        span_range = span_range.view(np.ndarray)

    index_of_leaves = visit_leaves_inorder(matrix)
    raw_worder = np.asarray([matrix[l, l] - 2 for l in
        index_of_leaves]).astype(np.intp)
    word_order = [vocab_size - 1 if w >= vocab_size else w for w in raw_worder]
    assert(all([w < vocab_size for w in word_order]))
    indices = sp.sparse.lil_matrix(((matrix.diagonal() > 0).sum(),
            vocab_size), dtype='uint8')
    for i, span in enumerate(span_range):
        word_size, word_counts = np.unique(word_order[slice(span[0], span[1])],
                return_counts=True)
        assert(np.all(word_counts < np.iinfo(np.uint8).max))
        if len(word_size) < span[1] - span[0]: # repeating words
            for j in range(len(word_size)): 
                indices[i, word_size[j]] += word_counts[j]
        else: # all unique words
            indices[i, word_size] = 1
    return(indices.tocsr())

def get_wordmatrix(matrix, embedding, is_to_norm=None):
    """
    return wordmatrix based on the tree matrix format
    """
    nodes_mask = np.ma.masked_where(matrix.diagonal() == 0,
            np.arange(matrix.shape[0]))
    input_vectors = np.zeros((matrix.shape[0], embedding.shape[1]))
    for inode in np.where(matrix.diagonal() > 1)[0]:
        input_vectors[inode] = embedding[int(matrix[inode, inode] - 2), :]

    if is_to_norm:
        input_vectors[nodes_mask.compressed()] = \
            normalize(input_vectors[nodes_mask.compressed()], norm=is_to_norm,
                axis=1, copy=False)
    return(input_vectors[nodes_mask.compressed()])


def gen_tree_building_confs(cur_len, sample_cutoff=100):
    """
    steps:
    1. acquire the total configurations of parsing tree given the size of
    leaves
    2. if the total number of configurations is less than sample_cutoff,
    using all tree configurations
    3. else choose the tree configurations within 95% confidence interval
    4. produce new samples for training
    >>> np.all(gen_tree_building_confs(2, sample_cutoff = 100) ==
    ... np.array([[1]]))
    True
    >>> np.all((np.eye(2) + 2*np.eye(2, k=1) + 2*np.eye(2, k =
    ... -1)).astype(np.intp) == gen_tree_building_confs(3, sample_cutoff =
    ... 100))
    True
    >>> len(gen_tree_building_confs(5, sample_cutoff = 10))
    10
    >>> len(gen_tree_building_confs(5, sample_cutoff = 100))
    24
    """
    is_exhaustive = False
    ttl_tree_confs = get_number_tree_configurations(cur_len)
    if ttl_tree_confs == 1:
        return(np.asarray([[1]], dtype=np.intp))
    elif ttl_tree_confs < sample_cutoff:
        is_exhaustive = True
    if is_exhaustive:
        return(np.asarray(list(permutations(range(1, cur_len)))))
    else:
        cur_hash_keys, cur_iter = {}, 0
        while(len(cur_hash_keys) < sample_cutoff):
            per_tree_conf = np.tile(np.arange(1, cur_len)[np.newaxis, :],
                                    (sample_cutoff - len(cur_hash_keys), 1))
            per_tree_conf = \
                np.asarray(list(map(np.random.permutation, per_tree_conf)))
            for arr in per_tree_conf:
                key = hash(arr.tobytes())
                if key not in cur_hash_keys:
                    cur_hash_keys[key] = arr
            if cur_iter > ttl_tree_confs:
                raise RuntimeError
        return(np.asarray([arr for arr in cur_hash_keys.values()],
            dtype=np.intp))


def preprocess_input(tree, vocab):
    """
    a utility function used to reformat input for theano compiled function
    @param tree is a nltk tree whose elemnts will be transformed into array
    @param vocab is the word vector
    """
    nterms = get_nonterminals(tree)
    n_leaves = len(tree.leaves())

    treeindex_map = OrderedDict([(idx, i) for i, idx in
        enumerate(sorted([tree[p]._index for p in tree.treepositions() if
            hasattr(tree[p], '_index')]))])

    mapping = np.repeat(np.arange(2 * n_leaves - 1), 2).reshape((-1, 2))
    wordmat = np.zeros((2 * n_leaves - 1, vocab.shape[1]))
    wordspan = np.ones((2 * n_leaves - 1, 2))

    for i, leaf in enumerate(list(tree.subtrees(lambda x: x.height() == 2))):
        wordmat[treeindex_map[leaf._index]] += vocab[i]

    traversal_order = np.empty(len(nterms), dtype=np.int)
    for i, (node_id, path_key) in enumerate(nterms):
        traversal_order[i] = treeindex_map[node_id]
        if isinstance(tree, Tree) and len(tree[path_key]) == 2:
            children_nodes = sorted(
                [tree[path_key][0], tree[path_key][1]], key=attrgetter(
                    '_index'))
            mapping[treeindex_map[node_id]] = [treeindex_map[node._index] for
                    node in children_nodes]
            wordspan[treeindex_map[node_id]] = [len(node.leaves()) for node in
                    children_nodes]
            assert(len(mapping[treeindex_map[node_id]]) == 2)
            assert(mapping[treeindex_map[node_id]][0] <
                    mapping[treeindex_map[node_id]][1])
    denom = np.sum(wordspan, axis=1)[:, np.newaxis]
    denom[denom == 0] = 1
    wordspan /= denom

    return(namedtuple("TreeInfo", "order mapping wordmat wordspan")._make(
        [traversal_order, mapping, wordmat, wordspan]))


# recursively deduce the parent index from path key
def encode_index(path, acc_num):
    if len(path) == 0:
        return(acc_num)
    else:
        acc_num = 2 * acc_num + path[0] + 1
        return(encode_index(path[1:], acc_num))


def two_passes_traversal(matrix, hook=None):
    """
    turn np.array or sp.sparse.* into nltk.tree.Tree from the following
    procedure:
    1. starting from all leaves
    2. merge one by one and counting the steps to root
    3. output the group according to the steps to the root from the greatest
       to the smallest (left-to-right order needs to be kept)
    the output can be handed to build_greedy_tree with build_order keyword to
    build nltk tree
    """
    root = 0 * (matrix.T.getrow(0).sum() == 1) + \
           (matrix.shape[0] - 1) * (
                   matrix.T.getrow(matrix.shape[0] - 1).sum() == 1)
    nodes_mask = np.ma.masked_where(matrix.diagonal() == 0,
            np.arange(matrix.shape[0]))
    if root == 0:  # taking the upper triangle offset 1
        out_edges = sp.sparse.triu(matrix, k=1)
    elif root == (matrix.shape[0] - 1):  # taking the lower triangle offset 1
        out_edges = sp.sparse.tril(matrix, k=-1)
    if out_edges.format != 'lil':
        out_edges = out_edges.tolil()

    cur_heights = np.zeros(matrix.shape[0], dtype=np.int)

    # bottom up traversal to update heights
    for n_iter in range(matrix.shape[0] // 2):
        cur_heights[(np.logical_and((out_edges.sum(axis=1) > 0).getA().ravel(),
            ~nodes_mask.mask))] += 1
        # removed links to current leaves
        out_edges[:, np.logical_and((out_edges.sum(axis=1) ==
            0).getA().ravel(), ~nodes_mask.mask)] = 0
        if np.all(out_edges.sum(axis=1) == 0):
            break

    # reset out_edges
    out_edges = sp.sparse.triu(matrix, k=1).tolil()

    # traverse top down to accumulate the heights to children
    cur_heights = topdown_traversal(matrix, cur_heights)
    # build_order as return the index of internal nodes ordered by heights
    non_terms = visit_inodes_inorder(matrix, root=root)
    build_order = np.lexsort((non_terms, -1 * cur_heights[non_terms]))

    if hook is not None:
        hook(build_order, cur_heights[non_terms])
    return(non_terms[build_order])


def topdown_traversal(matrix, cur_levels=None):
    """
    calculate the level to the root by top-down traversal
    """
    root = 0 * (matrix.T.getrow(0).sum() == 1) + \
           (matrix.shape[0] - 1) * (
                   matrix.T.getrow(matrix.shape[0] - 1).sum() == 1)
    if root == 0:  # taking the upper triangle offset 1
        out_edges = sp.sparse.triu(matrix, k=1)
    elif root == (matrix.shape[0] - 1):  # taking the lower triangle offset 1
        out_edges = sp.sparse.tril(matrix, k=-1)
    if out_edges.format != 'lil':
        out_edges = out_edges.tolil()

    is_reduce = False
    if cur_levels is None:
        cur_levels = np.ones(matrix.shape[0], dtype=np.int)
        is_reduce = True

    # traverse top down to accumulate the heights to children
    topdown_order, cur_groups = breadth_first_order(out_edges, root)
    topdown_order = deque(topdown_order)
    nodes = [topdown_order.popleft()]
    while len(topdown_order) > 0:
        for node in nodes[:]:
            cur_levels[np.logical_and((out_edges.sum(axis=0) >
                0).getA().ravel(), cur_groups == node)] += cur_levels[node]
        nodes = [topdown_order.popleft() for i in
                chain.from_iterable(map(lambda x: range(np.sum(cur_groups ==
                    x)), nodes))]
    if is_reduce:
        cur_levels -= np.ones(matrix.shape[0], dtype=np.int)
    return(cur_levels)


def is_heap_ordered(matrix):
    """
    retur True if the tree index is following a heap indexing scheme (holes
    will be introduced if the binary tree is incomplete); otherwise, return
    False
    """
    non_terms = visit_inodes_inorder(matrix)
    # firstly, checking if the root is at the first row
    if non_terms[-1] != 0:
        return(False)

    # secondly, picking the internal nodes and checking the index of their left
    # child is 2*i+1 and right child is 2*i+2
    if matrix.format != 'csr':
        matrix = matrix.tocsr()
    is_heapify = True
    for nt in non_terms:
        if not np.all(matrix.getrow(nt).indices ==
                [nt, 2 * nt + 1, 2 * nt + 2]):
            is_heapify = False
            break
    return(is_heapify)


def get_sentiment_label(pos_prob, num_labels=5,
        cutoff=np.arange(0.2, 1.2, 0.2)):

    if num_labels not in [2, 5]:
        raise ValueError('only accept 2 and 5 labels')
    if num_labels == 2:
        cutoff = cutoff[2::2]

    return(np.min(np.arange(num_labels, dtype=np.int)[
        float(pos_prob) <= cutoff]))
