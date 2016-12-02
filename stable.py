from __future__ import absolute_import
from __future__ import print_function

import os
import math
import random
import json
import collections

import numpy as np
import tensorflow as tf
from random import shuffle

# Set random seeds
SEED = 2016
random.seed(SEED)
np.random.seed(SEED)


# def build_dataset(filename, vocabulary_max=50000):
#     col_names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation',
#                  'relationship',
#                  'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country']
#
#     lines = [line.rstrip().split() for line in open(filename)]
#     docs_str = list()
#     wrds_str = list()
#     for line_no, line in enumerate(lines):
#         tokens = map("#".join, zip(col_names, line))
#         del (tokens[2])  # fnlwgt is meaningless
#         docs_str.append(tokens)
#         wrds_str.extend(tokens)
#
#     count = [['UNK', -1]]
#     count.extend(collections.Counter(wrds_str).most_common(vocabulary_max - 1))
#
#     dictionary = dict()
#     for word, _ in count:
#         dictionary[word] = len(dictionary)
#     reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
#
#     docs = list()
#     unk_count = 0
#     for doc_str in docs_str:
#         doc = list()
#         for word_str in doc_str:
#             if word_str in dictionary:
#                 index = dictionary[word_str]
#             else:
#                 index = 0  # dictionary['UNK']
#                 unk_count += 1
#             doc.append(index)
#         docs.append(doc)
#
#     count[0][1] = unk_count
#     return docs, count, dictionary, reverse_dictionary


def build_dataset(filename, vocabulary_max=50000):
    doc_ids = list()
    docs_str = list()
    wrds_str = list()
    for line in open(filename):
        doc_id, val = line.rstrip().split(':')
        wrds = val.split(',')
        doc_ids.append(doc_id)
        docs_str.append(wrds)
        wrds_str.extend(wrds)

    count = [['UNK', -1]]
    count.extend(collections.Counter(wrds_str).most_common(vocabulary_max - 1))

    dictionary = dict()
    for wrd, _ in count:
        dictionary[wrd] = len(dictionary)
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))

    docs = list()
    for doc_str in docs_str:
        doc = list()
        for wrd_str in doc_str:
            if wrd_str in dictionary:
                index = dictionary[wrd_str]
            else:
                index = 0  # dictionary['UNK']
            doc.append(index)
        docs.append(doc)

    return docs, doc_ids, dictionary, reverse_dictionary


eval_words = ['marital-status#Married-spouse-absent', 'education#5th-6th', 'workclass#State-gov',
              'native-country#Germany', 'occupation#Tech-support']


class Doc2Vec:
    def __init__(self, filename,
                 batch_size=1024,
                 doc_embed_dim=128,
                 wrd_embed_dim=128,
                 wrd_size_max=50000,
                 loss_type='sampled_softmax_loss',
                 optimizer_type='Adagrad',
                 learning_rate=1.0,
                 n_neg_samples=5,
                 n_steps=100001):

        self.docs, self.doc_ids, self.wrd_ids, self.reverse_wrd_ids = build_dataset(filename, wrd_size_max)
        self.doc_size = len(self.doc_ids)
        self.wrd_size = len(self.wrd_ids)
        print('doc size {}, word size {}'.format(self.doc_size, self.wrd_size))
        print('Sample doc: doc id {}, word id {}\n words {}'.format(self.doc_ids[0], self.docs[0],
                                                                    [self.reverse_wrd_ids[wrd] for wrd in
                                                                     self.docs[0]]))

        # bind params to class
        self.batch_size = batch_size
        self.doc_embed_dim = doc_embed_dim
        self.wrd_embed_dim = wrd_embed_dim
        self.loss_type = loss_type
        self.optimizer_type = optimizer_type

        self.learning_rate = learning_rate
        self.n_neg_samples = n_neg_samples
        self.n_steps = n_steps
        self.eval_examples = [self.wrd_ids[wrd] for wrd in eval_words]

        self._init_graph()
        self.sess = tf.Session(graph=self.graph)
        self.doc_idx = 0  # fetch training batch

    def _init_graph(self):
        self.graph = tf.Graph()
        with self.graph.as_default(), tf.device('/cpu:0'):
            # Set graph level random seed
            tf.set_random_seed(SEED)

            self.train_docs = tf.placeholder(tf.int32, shape=[self.batch_size])
            self.train_context = tf.placeholder(tf.int32, shape=[self.batch_size])
            self.train_labels = tf.placeholder(tf.int32, shape=[self.batch_size, 1])

            # Variables.
            self.wrd_embeddings = tf.Variable(
                tf.random_uniform([self.wrd_size, self.wrd_embed_dim], -1.0, 1.0))
            self.doc_embeddings = tf.Variable(
                tf.random_uniform([self.doc_size, self.doc_embed_dim], -1.0, 1.0))

            self.weights = tf.Variable(tf.truncated_normal([self.wrd_size, self.wrd_embed_dim + self.doc_embed_dim],
                                                           stddev=1.0 / math.sqrt(
                                                               self.wrd_embed_dim + self.doc_embed_dim)))
            self.biases = tf.Variable(tf.zeros([self.wrd_size]))

            # Embedding.
            wrd_embed = tf.nn.embedding_lookup(self.wrd_embeddings, self.train_context)
            doc_embed = tf.nn.embedding_lookup(self.doc_embeddings, self.train_docs)
            self.embed = tf.concat(1, [wrd_embed, doc_embed]) if self.doc_embed_dim > 0 else wrd_embed

            # Compute the loss, using a sample of the negative labels each time.
            loss = 0
            if self.loss_type == 'sampled_softmax_loss':
                loss = tf.nn.sampled_softmax_loss(self.weights, self.biases, self.embed, self.train_labels,
                                                  self.n_neg_samples, self.wrd_size)
            elif self.loss_type == 'nce_loss':
                loss = tf.nn.nce_loss(self.weights, self.biases, self.embed, self.train_labels,
                                      self.n_neg_samples, self.wrd_size)
            self.loss = tf.reduce_mean(loss)

            # Optimizer.
            if self.optimizer_type == 'Adagrad':
                self.optimizer = tf.train.AdagradOptimizer(self.learning_rate).minimize(self.loss)
            elif self.optimizer_type == 'SGD':
                self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss)

            # Similarity.
            norm = tf.sqrt(tf.reduce_sum(tf.square(self.wrd_embeddings), 1, keep_dims=True))
            normalized_embeddings = self.wrd_embeddings / norm
            valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, self.eval_examples)
            self.similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)

            # normalization
            # norm_w = tf.sqrt(tf.reduce_sum(tf.square(self.wrd_embeddings), 1, keep_dims=True))
            # self.normalized_word_embeddings = self.wrd_embeddings / norm_w
            #
            # norm_d = tf.sqrt(tf.reduce_sum(tf.square(self.doc_embeddings), 1, keep_dims=True))
            # self.normalized_doc_embeddings = self.doc_embeddings / norm_d

            # init op
            self.init_op = tf.initialize_all_variables()
            self.saver = tf.train.Saver()

    def fit(self):
        session = self.sess
        session.run(self.init_op)

        average_loss = 0
        print("Initialized")

        for step in xrange(self.n_steps):
            batch_docs, batch_context, batch_labels = self.generate_batch()
            feed_dict = {self.train_docs: batch_docs,
                         self.train_context: batch_context,
                         self.train_labels: batch_labels}

            _, l = session.run([self.optimizer, self.loss], feed_dict=feed_dict)

            # # debug
            # _, l, embed, dw_embed = session.run([self.optimizer, self.loss, self.embed, self.wrd_embeddings],
            #                                     feed_dict=feed_dict)
            # print('-----')
            # print('loss {}'.format(l))
            # print('length of embedding {}'.format(len(dw_embed)))
            # print(dw_embed)

            average_loss += l
            if step % 2000 == 0:
                if step > 0:
                    average_loss /= 2000
                # The average loss is an estimate of the loss over the last 2000 batches.
                print('Average loss at step %d: %f' % (step, average_loss))
                average_loss = 0

            if step % 10000 == 0:
                sim, wrd_embedding = session.run([self.similarity, self.wrd_embeddings], feed_dict=feed_dict)
                for i in xrange(len(self.eval_examples)):
                    valid_word = self.reverse_wrd_ids[self.eval_examples[i]]
                    embedding = wrd_embedding[self.eval_examples[i]]
                    top_k = 8  # number of nearest neighbors
                    nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                    log_str = "Nearest to %s:" % valid_word
                    for k in xrange(top_k):
                        close_word = self.reverse_wrd_ids[nearest[k]]
                        log_str = "%s %s," % (log_str, close_word)
                    print(log_str)
                    print(embedding)
        # update normalization
        # self.wrd_embeddings = session.run(self.normalized_word_embeddings)
        # self.doc_embeddings = session.run(self.normalized_doc_embeddings)
        return self

    def generate_batch(self):
        docs = np.ndarray(shape=self.batch_size, dtype=np.int32)
        context = np.ndarray(shape=self.batch_size, dtype=np.int32)
        labels = np.ndarray(shape=(self.batch_size, 1), dtype=np.int32)

        batch_idx = 0
        while True:
            doc = list(self.docs[self.doc_idx])
            ldoc = len(doc)
            shuffle(doc)
            for i in xrange(ldoc):
                docs[batch_idx] = self.doc_idx
                labels[batch_idx] = doc[i]
                context[batch_idx] = doc[(i + 1) % ldoc] if ldoc > 1 else 0
                batch_idx += 1
                if batch_idx == self.batch_size:
                    return docs, context, labels
            self.doc_idx = (self.doc_idx + 1) % self.doc_size

    def save(self, path):
        '''
        To save trained model and its params.
        '''
        save_path = self.saver.save(self.sess,
                                    os.path.join(path, 'model.ckpt'))
        # save parameters of the model
        params = self.get_params()
        json.dump(params,
                  open(os.path.join(path, 'model_params.json'), 'wb'))

        # save dictionary, reverse_dictionary
        json.dump(self.dictionary,
                  open(os.path.join(path, 'model_dict.json'), 'wb'),
                  ensure_ascii=False)
        json.dump(self.reverse_dictionary,
                  open(os.path.join(path, 'model_rdict.json'), 'wb'),
                  ensure_ascii=False)

        print("Model saved in file: %s" % save_path)
        return save_path

    def _restore(self, path):
        with self.graph.as_default():
            self.saver.restore(self.sess, path)

    @classmethod
    def restore(cls, path):
        '''
        To restore a saved model.
        '''
        # load params of the model
        path_dir = os.path.dirname(path)
        params = json.load(open(os.path.join(path_dir, 'model_params.json'), 'rb'))
        # init an instance of this class
        estimator = Doc2Vec(**params)
        estimator._restore(path)
        # evaluate the Variable embeddings and bind to estimator
        estimator.word_embeddings = estimator.sess.run(estimator.normalized_word_embeddings)
        estimator.doc_embeddings = estimator.sess.run(estimator.normalized_doc_embeddings)
        # bind dictionaries
        estimator.dictionary = json.load(open(os.path.join(path_dir, 'model_dict.json'), 'rb'))
        reverse_dictionary = json.load(open(os.path.join(path_dir, 'model_rdict.json'), 'rb'))
        # convert indices loaded from json back to int since json does not allow int as keys
        estimator.reverse_dictionary = {int(key): val for key, val in reverse_dictionary.items()}

        return estimator


a = Doc2Vec('adult_p.txt', n_steps=600001, doc_embed_dim=10, wrd_embed_dim=10)
a.fit()

# print('document_size {}'.format(a.document_size))
# print('vocabulary_size {}'.format(a.vocabulary_size))
#
# print(a.docs[0])
# print(a.docs[78])
#
# docs, context, labels = a.generate_batch()
# print('doc\n {} \n context\n {} \n labels\n {}'.format(docs, context, labels))
#
# print('length of batch: {}'.format(len(docs)))
#
# for idx in range(5):
#     print('---case {}:'.format(idx))
#     print(docs[idx])
#     print(context[idx])
#     print(labels[idx])
#     print('-----')
#
# print(a.docs[78])
# print('---case {}:'.format(-1))
# print(docs[-1])
# print(context[-1])
# print(labels[-1])
# print('-----')
