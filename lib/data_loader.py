import numpy as np
import itertools
from sklearn import cross_validation
import data_helpers_trained as data_helpers


def load_dataset(dataset_path, vocab_path):
    '''
    x: token ids;
    y: label one-hot encoding
    vocab: token to ids
    vocab_inv: id to tokens
    '''
    print "Loading data..."
    x, y, raw_y, vocabulary, vocabulary_inv, exist_idx = data_helpers.load_violence_data_binary(
        dataPath=dataset_path,
        wordPath=vocab_path)
    raw_y = np.array(raw_y)

    # Randomly shuffle data
    shuffle_indices = np.random.permutation(np.arange(len(y)))

    shuffled_x = x[shuffle_indices]
    shuffled_y = y[shuffle_indices]
    shuffled_raw_y = raw_y[shuffle_indices]

    print "ENTIRE DATASET --> none: {} violence: {} total: {}".format(len(raw_y[raw_y==0]), len(raw_y[raw_y==1]), len(raw_y))
    print "Vocabulary Size: {:d}".format(len(vocabulary))

    return shuffled_x, shuffled_y, shuffled_raw_y, vocabulary, vocabulary_inv, exist_idx


def get_WE_vectors(vectorPath, indices, vocabulary_inv, mode, msPath=None):
    ''' for the sake of saving memory, WE is partially loaded according to the words that aligned with the dataset '''
    print "loading pre-trained vectors..."
    we_vectors = np.load(vectorPath)
    we_vectors = we_vectors[indices]

    fill_vector = initialise_OOV(vocab_size=len(vocabulary_inv), we_vectors=we_vectors, msPath=msPath, mode=mode)
    trained_vectors = np.append(we_vectors, fill_vector, axis=0)
    print "word embedding size: {}".format(trained_vectors.shape)
    return trained_vectors


def initialise_OOV(vocab_size, we_vectors, msPath, mode):
    rows_diff = vocab_size - we_vectors.shape[0]
    if mode == "random" or mode == "normal":
        if not msPath:
            raise ValueError("embedding stats file not provided!")
    if mode == "zero":
        print "fill 0 for unseen words"
        fill_vector = np.zeros((rows_diff, we_vectors.shape[1]))
    elif mode =="random":
        print "randomly initializing unseen words..."
        ms = np.loadtxt(msPath, delimiter=',')
        means = ms[0]
        stds = ms[1]
        fill_vector = np.random.uniform(means-stds, means + stds, (rows_diff, we_vectors.shape[1]))
    elif mode == "pure":
        print "pure random strategy..."
        fill_vector = np.random.rand(rows_diff, we_vectors.shape[1])
    elif mode == "local":
        print "local random strategy..."
        means = np.mean(we_vectors, axis=0)
        stds = np.std(we_vectors,axis=0)
        fill_vector = np.random.uniform(means - stds, means + stds, (rows_diff, we_vectors.shape[1]))
    elif mode == "normal":
        print "normal strategy..."
        ms = np.loadtxt(msPath, delimiter=',')
        means = ms[0]
        stds = ms[1]
        fill_vector = np.random.normal(means, stds, (rows_diff, we_vectors.shape[1]))
    else:
        raise Exception("Wrong mode for random initialisation! option: [zero, random, pure, local]")
    return fill_vector


def k_cv_split(labels, k):
    folds = cross_validation.StratifiedKFold(labels, n_folds=k)
    n_folds = [n.tolist() for n in zip(*folds)[1]]
    cv_folds = build_folds(n_folds)
    return cv_folds


def build_folds(folds):
    ''' return index '''
    for i in xrange(len(folds)):
        validation = folds[i]
        try:
            test = folds[i+1]
            train = list(itertools.chain.from_iterable(folds[:i]+folds[i+2:]))
        except:
            test = folds[0]
            train = list(itertools.chain.from_iterable(folds[1:i]))
        yield train, validation, test


def test_we_loading(vocabulary_inv, trained_vectors, vectorPath, vocabPath):
    print "test vocabulary and vector"
    print vocabulary_inv[0]
    print trained_vectors[0][:20]
    print "============"
    print vocabulary_inv[100]
    print trained_vectors[100][:20]

    vocab = []
    with open(vocabPath, 'r') as f:
        for line in f:
            vocab.append(line.strip())
    we_vectors = np.load(vectorPath)

    idx1 = vocab.index(vocabulary_inv[0])
    idx2 = vocab.index(vocabulary_inv[100])

    print "real========"
    print vocab[idx1]
    print we_vectors[idx1][:20]
    print "==========="
    print vocab[idx2]
    print we_vectors[idx2][:20]
