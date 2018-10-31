import numpy as np
import math, os
import time
import shutil
import six
import os
#import psutil

import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector

import network, utils

class eni(object):
    def __init__(self, graph, args, sess):
        self.graph = graph
        self.args = args
        self.sess = sess
        self.degree_max = network.get_max_degree(self.graph)
        self.degree = network.get_degree(self.graph)
        self.save_path = os.path.join(self.args.save_path, '{}_{}_{}_{}'.format(self.args.save_suffix, self.args.embedding_size, self.args.alpha, self.args.lamb))

        self.build_model()

    def build_model(self):
        with tf.variable_scope('Placeholder'):
            self.nodes_placeholder = tf.placeholder(tf.int32, (None, ), name='nodes_placeholder')
            self.seqlen_placeholder = tf.placeholder(tf.int32, (None,), name='seqlen_placeholder')
            self.neighborhood_placeholder = tf.placeholder(tf.int32, (None, self.args.sampling_size), name='neighborhood_placeholder')
            self.label_placeholder = tf.placeholder(tf.float32, (None,), name='label_placeholder')

        self.data = network.next_batch(self.graph, self.degree_max, sampling=True, sampling_size=self.args.sampling_size)

        with tf.variable_scope('Embeddings'):
            self.embeddings = tf.get_variable('embeddings',
                    [len(self.graph), self.args.embedding_size],
                    initializer=tf.constant_initializer(utils.init_embedding(self.degree, self.degree_max, self.args.embedding_size)))

        with tf.variable_scope('LSTM'):
            cell = tf.contrib.rnn.DropoutWrapper(
                    #tf.contrib.rnn.BasicLSTMCell(num_units=self.args.embedding_size),
                    tf.contrib.rnn.LayerNormBasicLSTMCell(num_units=self.args.embedding_size, layer_norm=False),
                    input_keep_prob=1.0, output_keep_prob=1.0)
            _, states = tf.nn.dynamic_rnn(
                    cell,
                    tf.nn.embedding_lookup(self.embeddings, self.neighborhood_placeholder),
                    dtype=tf.float32,
                    sequence_length=self.seqlen_placeholder)
            self.lstm_output = states.h

        with tf.variable_scope('Guilded'):
            self.predict_info = tf.squeeze(tf.layers.dense(self.lstm_output, units=1, activation=utils.selu))


        with tf.variable_scope('Loss'):
            self.structure_loss = tf.losses.mean_squared_error(tf.nn.embedding_lookup(self.embeddings, self.nodes_placeholder), self.lstm_output)
            self.guilded_loss = tf.reduce_mean(tf.abs(tf.subtract(self.predict_info, self.label_placeholder)))
            self.orth_loss = tf.losses.mean_squared_error(tf.matmul(self.embeddings, self.embeddings, transpose_a=True), tf.eye(self.args.embedding_size))
            self.total_loss = self.structure_loss+self.args.alpha*self.orth_loss+self.args.lamb*self.guilded_loss

        with tf.variable_scope('Optimizer'):
            #self.optimizer = tf.train.AdamOptimizer(self.args.learning_rate)
            self.optimizer = tf.train.RMSPropOptimizer(self.args.learning_rate)
            tvars = tf.trainable_variables()
            grads, self.global_norm = tf.clip_by_global_norm(tf.gradients(self.total_loss, tvars), self.args.grad_clip)
            self.train_op = self.optimizer.apply_gradients(zip(grads, tvars))

        with tf.variable_scope('Summary'):
            tf.summary.scalar("orth_loss", self.orth_loss)
            tf.summary.scalar("guilded_loss", self.guilded_loss)
            tf.summary.scalar("structure_loss", self.structure_loss)
            tf.summary.scalar("total_loss", self.total_loss)
            tf.summary.scalar("globol_norm", self.global_norm)
            for (grad, var) in zip(grads, tvars):
                if grad is not None:
                    tf.summary.histogram('grad/{}'.format(var.name), grad)
                    tf.summary.histogram('weight/{}'.format(var.name), var)

            log_dir = os.path.join(self.save_path, 'logs')
            if os.path.exists(log_dir):
                shutil.rmtree(log_dir)
            self.summary_writer = tf.summary.FileWriter(log_dir, self.sess.graph)

            config = projector.ProjectorConfig()
            embedding = config.embeddings.add()
            embedding.tensor_name = self.embeddings.name
            embedding.metadata_path = os.path.join(os.path.join(self.args.save_path, 'data', 'index.tsv'))
            projector.visualize_embeddings(self.summary_writer, config)

            self.merged_summary = tf.summary.merge_all()

        self.saver = tf.train.Saver()
        self.sess.run(tf.global_variables_initializer())


    def fill_batch(self):
        inputs = [[] for _ in range(4)]
        for _ in range(self.args.batch_size):
            data, label = six.next(self.data)
            data += (label, )
            for input_, d in zip(inputs, data):
                input_.append(d)
        return {self.nodes_placeholder: inputs[0],
                self.neighborhood_placeholder: inputs[1],
                self.seqlen_placeholder: inputs[2],
                self.label_placeholder: inputs[3]}

    def get_embeddings(self):
        return self.embeddings.eval(session=self.sess)[1:]

    # @profile
    def train(self):
        print('training')
        total_num = int((len(self.graph)-1)/self.args.batch_size)
        if total_num < 16:
            self.args.batch_size = 2**int(np.log(len(self.graph)-1)/np.log(2)-2)
        total_num = int((len(self.graph)-1)/self.args.batch_size)
        print("batch_size: {}".format(self.args.batch_size))
        num = 0
        for epoch in range(self.args.epochs_to_train):
            orth_loss = 0.0
            guilded_loss = 0.0
            structure_loss = 0.0
            n = 0
            for i in range(total_num):
                begin = time.time()
                batch_data = self.fill_batch()
                _, total_loss, structure_loss, orth_loss, guilded_loss = self.sess.run([self.train_op, self.total_loss, self.structure_loss, self.orth_loss, self.guilded_loss], feed_dict=batch_data)
                n += 1
                end = time.time()
                #process = psutil.Process(os.getpid())
                print(("epoch: {}/{}, batch: {}/{}, loss: {:.6f}, structure_loss: {:.6f}, orth_loss: {:.6f}, guilded_loss: {:.6f}, time: {:.4f}s").format(epoch, self.args.epochs_to_train, n-1, total_num, total_loss, structure_loss, orth_loss, guilded_loss, end-begin))
                if num % 5 == 0:
                    summary_str = self.sess.run(self.merged_summary, feed_dict=batch_data)
                    self.summary_writer.add_summary(summary_str, num)
                num += 1
            self.save_model(epoch)
            if epoch % 10 == 0:
                self.save()

    def save_embeddings(self, save_path=None):
        print("Save embeddings in {}".format(save_path))
        embeddings = self.get_embeddings()
        filename = os.path.join(save_path, 'embeddings.npy')
        np.save(filename, embeddings)

    def save_model(self, step, name='eni'):
        save_path = self.save_path
        print("Save varibales in {}".format(save_path))
        self.saver.save(self.sess, os.path.join(save_path, 'eni'), global_step=step)

    def save(self, name='eni'):
        save_path = self.save_path
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        self.save_embeddings(save_path)
        with open(os.path.join(save_path, 'config.txt'), 'w') as f:
            for key, value in vars(self.args).items():
                if value is None:
                    continue
                if type(value) == list:
                    s_v = " ".join(list(map(str, value)))
                else:
                    s_v = str(value)
                f.write(key+" "+s_v+'\n')

