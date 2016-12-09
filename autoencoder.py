from __future__ import absolute_import
from __future__ import print_function

import random
import collections

import numpy as np
import tensorflow as tf

# Set random seeds
SEED = 20161206
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
    doc2id = dict()
    docs_str = list()
    wrds_str = list()
    for line in open(filename):
        doc_id, val = line.rstrip().split(':')
        wrds = val.split(',')
        doc2id[doc_id] = len(doc2id)
        docs_str.append(wrds)
        wrds_str.extend(wrds)

    count = [['UNK', -1]]
    count.extend(collections.Counter(wrds_str).most_common(vocabulary_max - 1))
    wrd2id = dict()
    for wrd, _ in count:
        wrd2id[wrd] = len(wrd2id)

    id2wrd = dict(zip(wrd2id.values(), wrd2id.keys()))
    assert len(wrd2id) == len(id2wrd)
    id2doc = dict(zip(doc2id.values(), doc2id.keys()))
    assert len(doc2id) == len(id2doc)

    dim = len(wrd2id)
    docs = list()
    for doc_str in docs_str:
        doc = np.zeros(dim, dtype=np.int32)
        for wrd_str in doc_str:
            idx = wrd2id[wrd_str] if wrd_str in wrd2id else 0  # dictionary['UNK']
            doc[idx] = 1  # binary or histogram?
        docs.append(doc)

    return docs, doc2id, id2doc, wrd2id, id2wrd


eval_words = ['marital-status#Married-spouse-absent', 'education#5th-6th', 'workclass#State-gov',
              'native-country#Germany', 'native-country#China', 'occupation#Tech-support']


class Autoencoder:
    def __init__(self, filename,
                 batch_size=1024,
                 embed_dim_1=128,
                 embed_dim_2=64,
                 wrd_size_max=10000,
                 learning_rate=0.01):

        self.docs, self.doc2id, self.id2doc, self.wrd2id, self.id2wrd = build_dataset(filename, wrd_size_max)
        self.doc_size = len(self.doc2id)
        self.wrd_size = len(self.wrd2id)
        print('doc size {}, word size {}'.format(self.doc_size, self.wrd_size))
        print('Sample doc: doc id {}, word {}\n total non-zeros {}'.format(self.id2doc[0], self.docs[0],
                                                                           np.sum(self.docs[0])))

        # bind params to class
        self.batch_size = batch_size
        self.embed_dim_1 = embed_dim_1
        self.embed_dim_2 = embed_dim_2
        self.learning_rate = learning_rate
        # self.eval_examples = np.random.choice(self.wrd_size, size=10, replace=False)
        self.eval_examples = [self.wrd2id[wrd] for wrd in eval_words]

        self._init_graph()
        self.sess = tf.Session(graph=self.graph)
        self.epoch = 0
        self.doc_idx = 0  # fetch training batch

    # Building the encoder
    def encoder(self, x):
        # Encoder Hidden layer with sigmoid activation #1
        layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, self.weights['encoder_h1']),
                                       self.biases['encoder_b1']))
        # Decoder Hidden layer with sigmoid activation #2
        layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, self.weights['encoder_h2']),
                                       self.biases['encoder_b2']))
        return layer_2

    # Building the decoder
    def decoder(self, x):
        # Encoder Hidden layer with sigmoid activation #1
        layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, self.weights['decoder_h1']),
                                       self.biases['decoder_b1']))
        # Decoder Hidden layer with sigmoid activation #2
        layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, self.weights['decoder_h2']),
                                       self.biases['decoder_b2']))
        return layer_2

    def _init_graph(self):
        self.graph = tf.Graph()
        with self.graph.as_default(), tf.device('/cpu:0'):
            # Set graph level random seed
            tf.set_random_seed(SEED)

            self.X = tf.placeholder("float", [self.batch_size, self.wrd_size])

            # Variables.
            self.weights = {
                'encoder_h1': tf.Variable(tf.random_normal([self.wrd_size, self.embed_dim_1])),
                'encoder_h2': tf.Variable(tf.random_normal([self.embed_dim_1, self.embed_dim_2])),
                'decoder_h1': tf.Variable(tf.random_normal([self.embed_dim_2, self.embed_dim_1])),
                'decoder_h2': tf.Variable(tf.random_normal([self.embed_dim_1, self.wrd_size])),
            }
            self.biases = {
                'encoder_b1': tf.Variable(tf.random_normal([self.embed_dim_1])),
                'encoder_b2': tf.Variable(tf.random_normal([self.embed_dim_2])),
                'decoder_b1': tf.Variable(tf.random_normal([self.embed_dim_1])),
                'decoder_b2': tf.Variable(tf.random_normal([self.wrd_size])),
            }

            self.code = self.encoder(self.X)
            self.Y = self.decoder(self.code)

            self.loss = tf.reduce_mean(tf.square(self.X - self.Y))
            self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)

            # Similarity.
            self.wrd_code = self.encoder(tf.constant(np.identity(self.wrd_size), dtype=tf.float32))
            norm = tf.sqrt(tf.reduce_sum(tf.square(self.wrd_code), 1, keep_dims=True))
            normalized_embeddings = self.wrd_code / norm
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

    def fit(self, n_epochs=10):
        print('Start model fitting (total {} epochs)'.format(n_epochs))
        session = self.sess
        session.run(self.init_op)

        step = 0
        average_loss = 0
        print("Initialized")

        while self.epoch < n_epochs:
            batch = self.generate_batch()
            _, l = session.run([self.optimizer, self.loss], feed_dict={self.X: batch})
            step += 1

            # # debug
            # _, l, embed, dw_embed = session.run([self.optimizer, self.loss, self.embed, self.wrd_embeddings],
            #                                     feed_dict=feed_dict)
            # print('-----')
            # print('loss {}'.format(l))
            # print('length of embedding {}'.format(len(dw_embed)))
            # print(dw_embed)

            average_loss += l
            if step % 2000 == 0:
                average_loss /= 2000
                # The average loss is an estimate of the loss over the last 2000 batches.
                print('Average loss at epoch %d step %d: %f' % (self.epoch, step, average_loss))
                average_loss = 0

            if step % 10000 == 0:
                sim, wrd_code, recode = session.run([self.similarity, self.wrd_code, self.Y], feed_dict={self.X: batch})
                print('error of the first case: {}'.format(sum(abs(batch[0] - recode[0]) > 0.01)))
                for i in xrange(len(self.eval_examples)):
                    valid_word = self.id2wrd[self.eval_examples[i]]
                    code = wrd_code[self.eval_examples[i]]
                    top_k = 8  # number of nearest neighbors
                    nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                    log_str = "Nearest to %s:" % valid_word
                    for k in xrange(top_k):
                        close_word = self.id2wrd[nearest[k]]
                        log_str = "%s %s," % (log_str, close_word)
                    print(log_str)
                    print(code)
        # update normalization
        # self.wrd_embeddings = session.run(self.normalized_word_embeddings)
        # self.doc_embeddings = session.run(self.normalized_doc_embeddings)
        return self

    def generate_batch(self):
        batch = np.ndarray(shape=(self.batch_size, self.wrd_size), dtype=np.int32)
        for i in xrange(self.batch_size):
            batch[i] = self.docs[self.doc_idx]
            self.doc_idx = (self.doc_idx + 1) % self.doc_size
            self.epoch += (self.doc_idx == 0)
        return batch

    def release(self, prefix=''):
        fdoc = open(prefix + '_doc.txt', 'w')
        doc_code = self.sess.run(self.code, feed_dict={self.X: self.docs})

        for i in xrange(self.doc_size):
            fdoc.write(self.id2doc[i] + ':')
            fdoc.write(",".join(map(str, doc_code[i])) + '\n')
        fdoc.close()


flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('input', 'adult_p.txt', 'input file')
flags.DEFINE_string('output', None, 'output files prefix')
flags.DEFINE_integer('embed_dim_1', 256, 'size of first hidden layer')
flags.DEFINE_integer('embed_dim_2', 128, 'size of second hidden layer')
flags.DEFINE_integer('n_epochs', 1000, 'number of epochs')
flags.DEFINE_integer('batch_size', 100, 'size of each batch')
flags.DEFINE_integer('wrd_size_max', 10000, 'size of word dictionary')

if FLAGS.output is None:
    FLAGS.output = FLAGS.input.split('.')[0] + '_ae'

a = Autoencoder(FLAGS.input, batch_size=FLAGS.batch_size, wrd_size_max=FLAGS.wrd_size_max,
                embed_dim_1=FLAGS.embed_dim_1, embed_dim_2=FLAGS.embed_dim_2)
a.fit(n_epochs=FLAGS.n_epochs)
a.release(prefix=FLAGS.output)
