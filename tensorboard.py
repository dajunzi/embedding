from __future__ import absolute_import
from __future__ import print_function

import os
import random
import json
import collections

import numpy as np
import tensorflow as tf
from random import shuffle

# Set random seeds
SEED = 20161202
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


def build_dataset(filename, vocabulary_max=10000):
    print('loading data...')
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
    del wrds_str

    wrd2id = dict()
    cat2id = collections.defaultdict(list)
    for wrd, _ in count:
        idx = len(wrd2id)
        wrd2id[wrd] = idx
        cat2id[wrd.split('#')[0]].append(idx)
    del cat2id['UNK']

    id2wrd = dict(zip(wrd2id.values(), wrd2id.keys()))
    assert len(wrd2id) == len(id2wrd)
    id2doc = dict(zip(doc2id.values(), doc2id.keys()))
    assert len(doc2id) == len(id2doc)

    docs = list()
    for doc_str in docs_str:
        doc = list()
        for wrd_str in doc_str:
            idx = wrd2id[wrd_str] if wrd_str in wrd2id else 0  # dictionary['UNK']
            doc.append(idx)
        docs.append(doc)

    return docs, doc2id, id2doc, wrd2id, id2wrd, cat2id


# eval_words = ['education#5th-6th', 'workclass#State-gov',
#               'native-country#Germany', 'native-country#China', 'occupation#Tech-support']


class Doc2Vec:
    def __init__(self, filename,
                 batch_size=1024,
                 doc_embed_dim=128,
                 wrd_embed_dim=128,
                 wrd_size_max=10000,
                 loss_type='sampled_softmax_loss',
                 optimizer_type='Adagrad',
                 learning_rate=1.0,
                 n_neg_samples=5,
                 eval_words=None):

        self.docs, self.doc2id, self.id2doc, self.wrd2id, self.id2wrd, self.cat2id = build_dataset(filename,
                                                                                                   wrd_size_max)
        self.doc_size = len(self.doc2id)
        self.wrd_size = len(self.wrd2id)

        # bind params to class
        self.batch_size = batch_size
        self.doc_embed_dim = doc_embed_dim
        self.wrd_embed_dim = wrd_embed_dim
        self.loss_type = loss_type
        self.optimizer_type = optimizer_type

        self.learning_rate = learning_rate
        self.n_neg_samples = n_neg_samples
        self.eval_examples = [self.wrd2id[wrd] for wrd in eval_words] if eval_words \
            else np.random.choice(self.wrd_size, size=10, replace=False)

        self._init_graph()
        self.sess = tf.Session(graph=self.graph)
        self.step = 0
        self.epoch = 0
        self.doc_idx = 0  # fetch training batch

        print('doc size {}, word size {}, doc dim {}, word dim {}'.format(self.doc_size, self.wrd_size,
                                                                          self.doc_embed_dim, self.wrd_embed_dim))
        print('Sample doc: doc id {}, word id {}\n words {}'.format(self.id2doc[0], self.docs[0],
                                                                    [self.id2wrd[wrd] for wrd in self.docs[0]]))

        # embedding projector
        meta_path = os.path.join(FLAGS.checkpoint_dir, 'metadata.tsv')
        with open(meta_path, 'w') as f:
            f.write('word\tlabel\n')
            for idx in xrange(self.wrd_size):
                f.write('{}\t{}\n'.format(self.id2wrd[idx], self.id2wrd[idx].split('#')[0]))

        from tensorflow.contrib.tensorboard.plugins import projector
        # Use the same LOG_DIR where you stored your checkpoint.
        summary_writer = tf.summary.FileWriter(FLAGS.checkpoint_dir)

        # Format: tensorflow/contrib/tensorboard/plugins/projector/projector_config.proto
        config = projector.ProjectorConfig()

        # You can add multiple embeddings. Here we add only one.
        embedding = config.embeddings.add()
        embedding.tensor_name = self.normalized.name
        embedding.metadata_path = meta_path

        # Saves a configuration file that TensorBoard will read during startup.
        projector.visualize_embeddings(summary_writer, config)
        # ----------

    def _init_graph(self):
        self.graph = tf.Graph()
        with self.graph.as_default(), tf.device('/cpu:0'):
            # Set graph level random seed
            tf.set_random_seed(SEED)

            self.train_docs = tf.placeholder(tf.int32, shape=[self.batch_size])
            self.train_context = tf.placeholder(tf.int32, shape=[self.batch_size])
            self.train_labels = tf.placeholder(tf.int32, shape=[self.batch_size, 1])

            # Variables.
            self.wrd_embeddings = tf.Variable(tf.random_uniform([self.wrd_size, self.wrd_embed_dim], -0.1, 0.1),
                                              name='wrd_embeddings')
            self.doc_embeddings = tf.Variable(tf.random_uniform([self.doc_size, self.doc_embed_dim], -0.1, 0.1),
                                              name='doc_embeddings')
            self.weights = tf.Variable(
                tf.random_uniform([self.wrd_size, self.wrd_embed_dim + self.doc_embed_dim], -0.1, 0.1), name='weights')
            self.biases = tf.Variable(tf.random_uniform([self.wrd_size], -0.1, 0.1), name='biases')
            self.normalized = tf.Variable(tf.zeros([self.wrd_size, self.wrd_embed_dim]), name='znormalized')

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

            # init op
            self.save_normalization = self.normalized.assign(normalized_embeddings)
            self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)
            self.init_op = tf.global_variables_initializer()

    def fit(self, n_epochs=10):
        print('Start model fitting (total {} epochs)'.format(n_epochs))
        session = self.sess
        session.run(self.init_op)

        average_loss = 0
        print("Initialized")

        while self.epoch < n_epochs:
            batch_docs, batch_context, batch_labels = self.generate_batch()
            feed_dict = {self.train_docs: batch_docs,
                         self.train_context: batch_context,
                         self.train_labels: batch_labels}

            _, l = session.run([self.optimizer, self.loss], feed_dict=feed_dict)
            self.step += 1

            # # debug
            # _, l, embed, dw_embed = session.run([self.optimizer, self.loss, self.embed, self.wrd_embeddings],
            #                                     feed_dict=feed_dict)
            # print('-----')
            # print('loss {}'.format(l))
            # print('length of embedding {}'.format(len(dw_embed)))
            # print(dw_embed)

            average_loss += l
            if self.step % 2000 == 0:
                average_loss /= 2000
                # The average loss is an estimate of the loss over the last 2000 batches.
                print('Average loss at epoch %d step %d: %f' % (self.epoch, self.step, average_loss))
                average_loss = 0

            if self.step % 20000 == 0:
                sim, _ = session.run([self.similarity, self.save_normalization], feed_dict=feed_dict)
                top_k = 5  # number of nearest neighbors

                for i in xrange(len(self.eval_examples)):
                    valid_word = self.id2wrd[self.eval_examples[i]]
                    print('----------------------------')
                    print('Nearest to {}:'.format(valid_word))

                    for cat in self.cat2id:
                        lst = self.cat2id[cat]
                        rng = min(top_k, len(lst))
                        log_str = ""
                        nearest = (-sim[i, lst]).argsort()[:rng]
                        for k in xrange(rng):
                            close_word = self.id2wrd[lst[nearest[k]]]
                            log_str = "%s %s," % (log_str, close_word)
                        print(log_str)

                self.save()

        return self

    def save(self):
        checkpoint_path = os.path.join(FLAGS.checkpoint_dir, 'model.ckpt')
        print('writing to {}'.format(checkpoint_path))
        self.saver.save(self.sess, checkpoint_path, global_step=self.step)

    def generate_batch(self):
        docids = np.ndarray(shape=self.batch_size, dtype=np.int32)
        context = np.ndarray(shape=self.batch_size, dtype=np.int32)
        labels = np.ndarray(shape=(self.batch_size, 1), dtype=np.int32)

        batch_idx = 0
        while True:
            self.doc_idx = (self.doc_idx + 1) % self.doc_size
            self.epoch += (self.doc_idx == 0)
            doc = list(self.docs[self.doc_idx])
            ldoc = len(doc)
            shuffle(doc)
            for i in xrange(ldoc):
                docids[batch_idx] = self.doc_idx
                labels[batch_idx] = doc[i]
                context[batch_idx] = doc[(i + 1) % ldoc] if ldoc > 1 else 0
                batch_idx += 1
                if batch_idx == self.batch_size:
                    return docids, context, labels

    def release(self, prefix=''):
        fwrd = open(prefix + '_wrd.txt', 'w')
        fdoc = open(prefix + '_doc.txt', 'w')

        wrd_embed, doc_embed = self.sess.run([self.wrd_embeddings, self.doc_embeddings])

        for i in xrange(self.wrd_size):
            fwrd.write(self.id2wrd[i] + ':')
            fwrd.write(",".join(map(str, wrd_embed[i])) + '\n')

        if self.doc_embed_dim > 0:
            for i in xrange(self.doc_size):
                fdoc.write(self.id2doc[i] + ':')
                fdoc.write(",".join(map(str, doc_embed[i])) + '\n')
        else:
            for i in xrange(self.doc_size):
                fdoc.write(self.id2doc[i] + ':')
                wrd_sum = np.zeros(self.wrd_embed_dim)
                for wrd in self.docs[i]:
                    wrd_sum += wrd_embed[wrd]
                wrd_sum /= len(self.docs[i])
                fdoc.write(",".join(map(str, wrd_sum)) + '\n')

        fwrd.close()
        fdoc.close()
        self.save()


print(tf.__version__)
flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('input', None, 'input file')
flags.DEFINE_string('output', None, 'output files prefix')
flags.DEFINE_string('config', None, 'config file')
flags.DEFINE_string('checkpoint_dir', '/tmp/data', 'Directory to store checkpoint data.')
flags.DEFINE_integer('doc_embed_dim', 0, 'document embedding size')
flags.DEFINE_integer('wrd_embed_dim', 64, 'word embedding size')
flags.DEFINE_integer('n_epochs', 100, 'number of epochs')
flags.DEFINE_integer('batch_size', 2000, 'size of each batch')
flags.DEFINE_integer('wrd_size_max', 10000, 'max size of vocabulary')

version = 'wrd2vec' if FLAGS.doc_embed_dim == 0 else 'doc2vec'
print(version, 'embedding')

if FLAGS.output is None:
    FLAGS.output = FLAGS.input.split('.')[0] + '_' + version

eval_words = [line.rstrip() for line in open(FLAGS.config)] if FLAGS.config else None

a = Doc2Vec(FLAGS.input, batch_size=FLAGS.batch_size, doc_embed_dim=FLAGS.doc_embed_dim,
            wrd_embed_dim=FLAGS.wrd_embed_dim, wrd_size_max=FLAGS.wrd_size_max, eval_words=eval_words)
a.fit(n_epochs=FLAGS.n_epochs)
a.release(prefix=FLAGS.output)
