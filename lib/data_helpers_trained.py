import numpy as np
import re
import itertools
from collections import Counter
import csv

"""
Codes are adapted from https://github.com/dennybritz/cnn-text-classification-tf 
"""

# def clean_str(string):
#     """
#     Tokenization/string cleaning for all datasets except for SST.
#     Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
#     """
#     string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
#     string = re.sub(r"\'s", " \'s", string)
#     string = re.sub(r"\'ve", " \'ve", string)
#     string = re.sub(r"n\'t", " n\'t", string)
#     string = re.sub(r"\'re", " \'re", string)
#     string = re.sub(r"\'d", " \'d", string)
#     string = re.sub(r"\'ll", " \'ll", string)
#     string = re.sub(r",", " , ", string)
#     string = re.sub(r"!", " ! ", string)
#     string = re.sub(r"\(", " \( ", string)
#     string = re.sub(r"\)", " \) ", string)
#     string = re.sub(r"\?", " \? ", string)
#     string = re.sub(r"\s{2,}", " ", string)
#     return string.strip().lower()

def load_data_and_labels(dataPath):
    """
    Load dataset from files, splits the data into text and generate labels.
    Labels are binary.
    """
    # Load data from files
    with open(dataPath, 'r') as f:
        reader = csv.reader(f)
        raw_data = [line for line in reader]
    label_map = {'0':[1,0], '1': [0,1]}
    x_text = [line[1:] for line in raw_data]
    y = [label_map[line[0]] for line in raw_data]
    raw_y = [int(line[0]) for line in raw_data]
    # print y

    return [x_text, y, raw_y]

def load_data_and_labels_violence(dataPath):
    """
    Load dataset from files, splits the data into text and generates labels.
    Labels are not binary.
    """
    # Load data from files
    with open(dataPath, 'r') as f:
        reader = csv.reader(f)
        raw_data = [line for line in reader]
    label_map = {'0':[1,0,0], '1': [0,1,0], '2': [0,0,1]}
    x_text = [line[1:] for line in raw_data]
    y = [label_map[line[0]] for line in raw_data]
    raw_y = [int(line[0]) for line in raw_data]
    # print y

    return [x_text, y, raw_y]

def load_data_and_labels_violence_binary(dataPath):
    """
    Load dataset from files, splits the data into text and generates labels.
    Labels are binary and set malpractice as not violence
    """
    # Load data from files
    with open(dataPath, 'r') as f:
        reader = csv.reader(f)
        raw_data = [line for line in reader]
    label_map = {'0':[1,0], '1': [0,1], '2': [1,0]}
    binary_map = {'0': 0, '1': 1, '2': 0}
    x_text = [line[1:] for line in raw_data]
    y = [label_map[line[0]] for line in raw_data]
    raw_y = [binary_map[line[0]] for line in raw_data]
    # print y

    return [x_text, y, raw_y]

def pad_sentences(sentences, padding_word="<PAD/>"):
    """
    Pads all sentences to the same length. The length is defined by the longest sentence.
    Returns padded sentences.
    """
    sequence_length = max(len(x) for x in sentences)
    padded_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        num_padding = sequence_length - len(sentence)
        new_sentence = sentence + [padding_word] * num_padding
        padded_sentences.append(new_sentence)
    return padded_sentences


def build_vocab(sentences,wordPath):
    """
    Builds a vocabulary mapping from word to index based on the sentences.
    Returns vocabulary mapping and inverse vocabulary mapping.
    """
    #words of pre-trained WE
    words = {}
    WE_words = []

    # wordPath = 'trained_words.txt'
    with open(wordPath, 'r') as f:
        count = 0
        for line in f:
            aword = line.strip()
            words[aword] = count
            WE_words.append(aword)
            count += 1
    print "pretrained words length: {}".format(len(words))

    # Build vocabulary
    word_counts = Counter(itertools.chain(*sentences))
    # Mapping from index to word
    data_words = zip(*word_counts.most_common())[0]
    print "words appear in data set: {}".format(len(data_words))

    exist_idx = [words[w] for w in data_words if w in words]
    # exist_idx = [WE_words.index(w) for w in data_words if w in WE_words] # for test only

    print "words appear in pretrained WE: {}".format(len(exist_idx))
    print "words coverage: {}%".format(len(exist_idx)*100/len(data_words))
    # print exist_idx
    exist_idx = sorted(exist_idx)
    # print exist_idx
    exist_words = [WE_words[i] for i in exist_idx]

    newwords = [w for w in data_words if w not in words]
    print "words not appear in pretrained WE: {}".format(len(newwords))
    vocabulary_inv = exist_words + newwords
    # Mapping from word to index
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
    return [vocabulary, vocabulary_inv, exist_idx]

def build_input_data(sentences, labels, vocabulary):
    """
    Maps sentencs and labels to vectors based on a vocabulary.
    """
    x = np.array([[vocabulary[word] for word in sentence] for sentence in sentences])
    y = np.array(labels)
    return [x, y]

def get_WE_length(wordPath):
    count = 0
    # wordPath = 'trained_words.txt'
    with open(wordPath, 'r') as f:
        for line in f:
            count += 1
    return count

def load_data(dataPath, wordPath):
    """
    Load the preprocessed data for the dataset.
    Labels are binary.
    """
    # Load and preprocess data
    sentences, labels, raw_y = load_data_and_labels(dataPath=dataPath)
    sentences_padded = pad_sentences(sentences)
    print "padded sentence length : {}".format(len(sentences_padded[0]))

    vocabulary, vocabulary_inv, exist_idx = build_vocab(sentences=sentences_padded, wordPath=wordPath)
    x, y = build_input_data(sentences_padded, labels, vocabulary)
    return [x, y, raw_y, vocabulary, vocabulary_inv, exist_idx]

def load_violence_data(dataPath, wordPath):
    """
    Load the preprocessed data for the dataset.
    Labels are not binary.
    """
    # Load and preprocess data
    sentences, labels, raw_y = load_data_and_labels_violence(dataPath=dataPath)
    sentences_padded = pad_sentences(sentences)
    print "padded sentence length : {}".format(len(sentences_padded[0]))

    vocabulary, vocabulary_inv, exist_idx = build_vocab(sentences=sentences_padded, wordPath=wordPath)
    x, y = build_input_data(sentences_padded, labels, vocabulary)
    return [x, y, raw_y, vocabulary, vocabulary_inv, exist_idx]

def load_violence_data_binary(dataPath, wordPath):
    """
    Load thepreprocessed data for the dataset.
    Labels are binary and malpractice is set as not violence
    """
    # Load and preprocess data
    sentences, labels, raw_y = load_data_and_labels_violence_binary(dataPath=dataPath)
    sentences_padded = pad_sentences(sentences)
    print "padded sentence length : {}".format(len(sentences_padded[0]))

    vocabulary, vocabulary_inv, exist_idx = build_vocab(sentences=sentences_padded, wordPath=wordPath)
    x, y = build_input_data(sentences_padded, labels, vocabulary)
    return [x, y, raw_y, vocabulary, vocabulary_inv, exist_idx]

def batch_iter(data, batch_size, num_epochs):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    if len(data) % batch_size == 0:
        num_batches_per_epoch = int(len(data) / batch_size)
    else:
        num_batches_per_epoch = int(len(data) / batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_data = data[shuffle_indices]
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]

if __name__ == '__main__':
    wordPath = 'model_w5_d500.words'
    dataPath = 'election_tokens.csv'
    x,y,v, vinx= load_data(dataPath=dataPath, wordPath=wordPath)
    print len(x)
    print len(y)
    print v['infrar']
    print len(vinx)