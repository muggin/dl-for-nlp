from __future__ import division

import os
import re
import math
import time
import nltk
import string
import random
import codecs
import numpy as np
import itertools as it
import tensorflow as tf
import cPickle as pickle
import scipy.sparse as ss
import matplotlib.pyplot as plt

from tqdm import tqdm
from collections import Counter
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

def load_corpus(file_path):
    """ Load corpus from text file and tokenize """
    corpus = []
    vocab_cnt = Counter()
    tokenizer = nltk.tokenize.WhitespaceTokenizer()

    with codecs.open(file_path, 'r', encoding='utf-8') as fd:
        for line in fd:
            # clean lines from any punctuation characters
            clean_line = re.sub('[\+\-\.\,\:\;\"\?\!\>\<\=\(\)\n]+', '', line)
            tokens = tokenizer.tokenize(clean_line.lower())
            corpus.append(tokens)
            vocab_cnt.update(tokens)

    return corpus, vocab_cnt


def code_tokens(vocab_cnt, max_size=30000, unk_symbol='<unk>'):
    """ Filter vocabulary and encode tokens """
    vocab = ['<pad>', '<sos>', '<eos>', '<unk>']
    vocab.extend([word for word, _ in vocab_cnt.most_common(max_size)])
    vocab_enc = {token: ix for ix, token in enumerate(vocab)}
    vocab_dec = {ix: token for token, ix in vocab_enc.iteritems()}

    return vocab, vocab_enc, vocab_dec


def generate_context_data(corpus, max_window_size=5, skip_size=1, flatten=True):
    """ Generate data with context in format (target, [contexts]) or (target, context) """
    for center_ix in xrange(max_window_size, len(corpus)-max_window_size, skip_size):
        # sample a window size for the given center word
        window_size = np.random.randint(max_window_size) + 1
        full_context = corpus[center_ix-window_size:center_ix] + corpus[center_ix+1: center_ix+window_size+1]

        if flatten:
            for context_ix in xrange(2*window_size):
                yield (corpus[center_ix], full_context[context_ix])
        else:
            yield(corpus[center_ix], full_context)


def pad_data(data_arr, append_pre=[], append_suf=[], max_length=None):
    """ Pad sequences to length of longest sequence in batch. Possibly append or prepend tokens """
    data_arr = [append_pre + row + append_suf for row in data_arr]
    lengths = [len(row) for row in data_arr]
    max_len = max(lengths) if not max_length else max_length
    return np.array([row+[0]*(max_len-length) for row, length in zip(data_arr, lengths)]), lengths


def batchify_data(data_generator, batch_size):
    """ Split dataset (generator) into batches """
    if isinstance(data_generator, list):
        for ix in xrange(0, len(data_generator), batch_size):
            buff = data_generator[ix:ix+batch_size]
            yield buff
    else:
        while data_generator:
            buff = []
            for ix in xrange(0, batch_size):
                buff.append(next(data_generator))
            yield buff


def toy_data_generator(vocab_size, data_size, max_seq_length, reserved_digits=3):
    """ Generate toy data of integers up to Vocab Size """
    for _ in xrange(data_size):
        seq_length = np.random.randint(max_seq_length) + 1
        output = [np.random.randint(vocab_size-reserved_digits)+reserved_digits for _ in xrange(seq_length)]
        yield (output, output)


def save_embeddings(embeddings_obj, file_name):
    """ Save word embeddings and helper structures """
    with open(file_name, 'wb') as fd:
        pickle.dump(embeddings_obj, fd)


def load_embeddings(file_name):
    """ Load word embeddings and helper structures """
    with open(file_name, 'r') as fd:
        embeddings_obj = pickle.load(fd)
    return embeddings_obj


def get_tsne_embeddings(embedding_matrix):
    """ Compute t-SNE representation of embeddings """
    tsne = TSNE(perplexity=25, n_components=2, init='pca', n_iter=5000)
    return tsne.fit_transform(embedding_matrix)


def get_pca_embeddings(embedding_matrix):
    """ Compute PCA representation of embeddings """
    pca = PCA(n_components=2)
    return pca.fit_transform(embedding_matrix)


def plot_embeddings(embeddings, words=[], words_cnt=500, method='pca', figsize=(8,8)):
    """ Plot subset of embeddings in 2D space using t-SNE or PCA """
    embedding_matrix = embeddings._embeddings
    vocab_dec = embeddings._vocab_dec
    vocab_enc = embeddings._vocab_enc

    # prepare data
    if not words:
        vocab_size = embedding_matrix.shape[0]
        ixs = range(vocab_size)
        random.shuffle(ixs)
        chosen_ixs = ixs[:words_cnt]
        labels = [vocab_dec[ix] for ix in chosen_ixs]
        word_vecs = embedding_matrix[chosen_ixs]
    else:
        labels = words
        chosen_ixs = [vocab_enc[word] for word in words]
        word_vecs = embedding_matrix[chosen_ixs]

    if method == 'tsne':
        low_dim_embeddings = get_tsne_embeddings(word_vecs)
    else:
        low_dim_embeddings = get_pca_embeddings(word_vecs)

    # plot reduced vectors
    plt.figure(figsize=figsize)

    for embedding, label in zip(low_dim_embeddings, labels):
        x, y = embedding[0], embedding[1]
        plt.scatter(x, y)
        plt.annotate(label, xy=(x, y), xytext=(5, 2),
                     textcoords='offset points', ha='right',
                     va='bottom')
    plt.yticks([])
    plt.xticks([])
    plt.grid()
    plt.show()


def plot_alignment(attn_img, source_seq, target_seq, figsize=(8,8)):
    """ Plot alignment matrix with source, target labels """
    plt.figure(figsize=figsize)
    plt.imshow(attn_img, cmap='gray', interpolation='none')

    plt.gca().set_xticks(np.arange(0, len(target_seq), 1))
    plt.gca().set_xticklabels(list(target_seq), rotation=45)
    plt.gca().tick_params(labelbottom='off',labeltop='on')

    plt.gca().set_yticks(np.arange(0, len(source_seq), 1))
    plt.gca().set_yticklabels(list(source_seq))
    plt.show()


class Embeddings(object):
    """ Class wrapping word embeddings """
    def __init__(self, embedding_matrix, vocab_enc, vocab_dec):
        self._embeddings = embedding_matrix
        self._vocab_enc = vocab_enc
        self._vocab_dec = vocab_dec

    def find_embedding(self, word):
        """ Find embedding for a given word """
        if isinstance(word, str):
            word = self._vocab_enc[word]
        return self._embeddings[word]

    def find_neighbors(self, word, k=5, nearest=True, exclude=[], include_scores=False):
        """ Find neighboring words (semantic regularities) """
        word_ix = self._vocab_enc[word]
        exclude = exclude + [word_ix]

        # find neighbors
        word_emb = self._embeddings[word_ix]
        similarities = self._embeddings.dot(word_emb)
        similarities[exclude] = 0
        best_matches = np.argsort(similarities)
        trimmed_matches = best_matches[-k:][::-1] if nearest else best_matches[:k]
        return [(self._vocab_dec[word_ix], similarities[word_ix]) for word_ix in trimmed_matches]

    def find_analogous(self, word_a, word_b, word_c, k=5):
        """ Find analogous word (syntactic regularities: word_a - word_b = x - word_c) """
        word_a_ix, word_b_ix, word_c_ix = [self._vocab_enc[word] for word in [word_a, word_b, word_c]]
        exclude = [word_a_ix, word_b_ix, word_c_ix]

        emb_a = self.find_embedding(word_a_ix)
        emb_b = self.find_embedding(word_b_ix)
        emb_c = self.find_embedding(word_c_ix)
        emb_d_hat = emb_a - emb_b + emb_c
        similarities = self._embeddings.dot(emb_d_hat)
        similarities[exclude] = 0
        best_matches = np.argsort(similarities)
        trimmed_matches = best_matches[-k:][::-1]
        return [(self._vocab_dec[word_ix], similarities[word_ix]) for word_ix in trimmed_matches]

    def vocab(self):
        """ Return vocabulary list """
        return self._vocab_enc.keys()
