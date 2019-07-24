import tensorflow as tf
import numpy as np
from my_lstm import MyLSTM
from helper import *
from tensorflow.python.ops import rnn_cell_impl
#linear = rnn_cell_impl._linear
linear = tf.contrib.layers.fully_connected
LSTMStateTuple = tf.nn.rnn_cell.LSTMStateTuple


class JMT:

    def __init__(self, dim, reg_lambda, lr=0.01):
        self.dim = dim
        self.reg_lambda = reg_lambda
        self.lr = lr

    def load_data(self):
        data = np.load('data/data.npz', allow_pickle=True)['data'].item()
        self.sent = data['word_level']['sent'].values
        self.pos = data['word_level']['pos']
        self.i2p = data['word_level']['i2p']
        self.i2c = data['word_level']['i2c']
        self.chun = data['word_level']['chunk']
        self.sent1 = data['sent_level']['sent1']
        self.sent2 = data['sent_level']['sent2']
        self.i2e = data['sent_level']['i2e']
        self.rel = data['sent_level']['rel']
        self.ent = data['sent_level']['entailment']
        self.w2i = data['w2i']

        self.vec = np.array(data['vec'] + [[0] * 300])
        self.max_length = max([len(i) for i in self.sent])
        print('***Data loaded***')

    def build_model(self):
        ''' Builds the whole computational graph '''
        def sentence_op(inputs, t_pos, t_chunk):
            with tf.variable_scope('pos',reuse=tf.AUTO_REUSE):
                embeddings = tf.constant(self.vec, dtype=tf.float32)
                embeds = tf.nn.embedding_lookup(embeddings, inputs)
                fw_lstm = MyLSTM(self.dim, state_is_tuple=True)
                bw_lstm = MyLSTM(self.dim, state_is_tuple=True)
                outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=fw_lstm, cell_bw=bw_lstm, inputs=embeds,
                                                             sequence_length=length(embeds), dtype=tf.float32)
                concat_outputs = tf.concat(outputs,2)
                y_pos = activate(concat_outputs, [
                    self.dim * 2, len(self.i2p)], [len(self.i2p)])
                t_pos_sparse = tf.one_hot(
                    indices=t_pos, depth=len(self.i2p), axis=-1)
                loss = cost(y_pos, t_pos_sparse)
                loss += tf.reduce_sum([self.reg_lambda * tf.nn.l2_loss(x)
                                       for x in tf.trainable_variables()])
                optimize_op = tf.train.AdagradOptimizer(
                    self.lr).minimize(loss)

            with tf.variable_scope('chunk',reuse=tf.AUTO_REUSE):
                inputs1 = tf.concat([embeds, concat_outputs, y_pos],2)
                fw_lstm = MyLSTM(self.dim, state_is_tuple=True)
                bw_lstm = MyLSTM(self.dim, state_is_tuple=True)
                outputs1, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=fw_lstm, cell_bw=bw_lstm, inputs=inputs1,
                                                              sequence_length=length(embeds), dtype=tf.float32)
                concat_outputs1 = tf.concat(outputs1,2)
                y_chunk = activate(concat_outputs1, [
                                   self.dim * 2, len(self.i2c)], [len(self.i2c)])
                t_chunk_sparse = tf.one_hot(
                    indices=t_chunk, depth=len(self.i2c), axis=-1)
                loss1 = cost(y_chunk, t_chunk_sparse)
                loss1 += tf.reduce_sum([self.reg_lambda * tf.nn.l2_loss(x)
                                        for x in tf.trainable_variables()])
                optimize_op1 = tf.train.AdagradOptimizer(
                    self.lr).minimize(loss1)

            with tf.variable_scope('relatedness'):
                with tf.variable_scope('layer_1'):
                    inputs2 = tf.concat(
                         [embeds, concat_outputs1, y_pos, y_chunk],2)
                    fw_lstm = MyLSTM(self.dim, state_is_tuple=True)
                    bw_lstm = MyLSTM(self.dim, state_is_tuple=True)
                    outputs2, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=fw_lstm, cell_bw=bw_lstm, inputs=inputs2,
                                                                  sequence_length=length(embeds), dtype=tf.float32)
                    concat_outputs2 = tf.concat(outputs2,2)
                with tf.variable_scope('layer_2'):
                    inputs3 = tf.concat(
                        [embeds, concat_outputs2, y_pos, y_chunk],2)
                    fw_lstm1 = MyLSTM(self.dim, state_is_tuple=True)
                    bw_lstm1 = MyLSTM(self.dim, state_is_tuple=True)
                    outputs3, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=fw_lstm1, cell_bw=bw_lstm1, inputs=inputs3,
                                                                  sequence_length=length(embeds), dtype=tf.float32)
                    concat_outputs3 = tf.concat(outputs3,2)
                    s = tf.reduce_max(concat_outputs3, reduction_indices=1)

                with tf.variable_scope('layer_3'):
                    inputs4 = tf.concat(
                        [embeds, concat_outputs3, y_pos, y_chunk],2)
                    fw_lstm2 = MyLSTM(self.dim, state_is_tuple=True)
                    bw_lstm2 = MyLSTM(self.dim, state_is_tuple=True)
                    outputs4, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=fw_lstm2, cell_bw=bw_lstm2, inputs=inputs4,
                                                                  sequence_length=length(embeds), dtype=tf.float32)
                    concat_outputs4 = tf.concat(outputs3,2)
                    s1 = tf.reduce_max(concat_outputs4, reduction_indices=1)

            return s, s1, optimize_op, optimize_op1, loss, loss1, y_pos, y_chunk

        with tf.variable_scope('sentence') as scope:
            self.inp = tf.placeholder(
                shape=[None, self.max_length], dtype=tf.int32, name='input')
            self.t_p = tf.placeholder(
                shape=[None, self.max_length], dtype=tf.int32, name='t_pos')
            self.t_c = tf.placeholder(
                shape=[None, self.max_length], dtype=tf.int32, name='t_chunk')
            s11, s12, self.optimize_op, self.optimize_op1, self.loss, self.loss1, self.y_pos, self.y_chunk = sentence_op(
                self.inp, self.t_p, self.t_c)
            scope.reuse_variables()
            self.inp1 = tf.placeholder(
                shape=[None, self.max_length], dtype=tf.int32, name='input1')
            s21, s22 = sentence_op(self.inp1, self.t_p, self.t_c)[:2]

            d = tf.concat([tf.abs(tf.subtract(s11, s21)), tf.multiply(s11, s21)],1)
            d1 = tf.concat([tf.subtract(s12, s22), tf.multiply(s12, s22)],1)
        with tf.variable_scope('relation'):
            self.y_rel = tf.squeeze(
                activate(d, [self.dim * 4, 1], [1], activation=tf.nn.relu))
            self.t_rel = tf.placeholder(shape=[None], dtype=tf.float32)
            self.loss2 = rmse_loss(self.y_rel, self.t_rel)
            self.loss2 += tf.reduce_sum([self.reg_lambda * tf.nn.l2_loss(x)
                                         for x in tf.trainable_variables()])
            self.optimize_op2 = tf.train.AdagradOptimizer(
                self.lr).minimize(self.loss2)

        with tf.variable_scope('entailment'):
            self.t_ent = tf.placeholder(shape=[None], dtype=tf.int32)
            t_ent_sparse = tf.one_hot(indices=self.t_ent, depth=3, axis=-1)
            self.y_ent = activate(d1, [self.dim * 4, 3], [3])
            self.loss3 = - tf.reduce_mean(t_ent_sparse * tf.log(self.y_ent))
            self.loss3 += tf.reduce_sum([self.reg_lambda * tf.nn.l2_loss(x)
                                         for x in tf.trainable_variables()])
            self.optimize_op3 = tf.train.AdagradOptimizer(
                self.lr).minimize(self.loss3)
        print('***Model built***')

    def get_predictions(self, graph, task_desc):
        resp = dict()
        saver = tf.train.Saver()
        with tf.Session(graph=graph) as sess:
            saver = tf.train.import_meta_graph('saves/model.ckpt.meta')
            saver.restore(sess, tf.train.latest_checkpoint('./saves'))

            if 'pos' in task_desc:
                inp = task_desc['pos'].lower().split()
                inputs = [[self.w2i[i] for i in inp] +
                          [self.vec.shape[0] - 1] * (self.max_length - len(inp))]
                preds = sess.run(self.y_pos,
                                 {self.inp: inputs})[0]
                preds = np.argmax(preds, axis=-1)[:len(inp)]
                preds = [self.i2p[i] for i in preds]
                resp['pos'] = preds

            if 'chunk' in task_desc:
                inp = task_desc['chunk'].lower().split()
                inputs = [[self.w2i[i] for i in inp] +
                          [self.vec.shape[0] - 1] * (self.max_length - len(inp))]
                preds = sess.run(self.y_chunk, {self.inp: inputs})[0]
                preds = np.argmax(preds, axis=-1)[:len(inp)]
                preds = [self.i2c[i] for i in preds]
                resp['chunk'] = preds

            if 'relatedness' in task_desc:
                inp1 = task_desc['relatedness'][0].lower().split()
                inputs1 = [[self.w2i[i] for i in inp1] +
                           [self.vec.shape[0] - 1] * (self.max_length - len(inp1))]
                inp2 = task_desc['relatedness'][1].lower().split()
                inputs2 = [[self.w2i[i] for i in inp2] +
                           [self.vec.shape[0] - 1] * (self.max_length - len(inp2))]
                preds = sess.run(
                    self.y_rel, {self.inp: inputs1, self.inp1: inputs2})
                resp['relatedness'] = preds

            if 'entailment' in task_desc:
                inp1 = task_desc['entailment'][0].lower().split()
                inputs1 = [[self.w2i[i] for i in inp1] +
                           [self.vec.shape[0] - 1] * (self.max_length - len(inp1))]
                inp2 = task_desc['entailment'][1].lower().split()
                inputs2 = [[self.w2i[i] for i in inp2] +
                           [self.vec.shape[0] - 1] * (self.max_length - len(inp2))]
                preds = sess.run(
                    self.y_ent, {self.inp: inputs1, self.inp1: inputs2})[0]
                resp['entailment'] = self.i2e[np.argmax(preds)]

        return resp

    def train_model(self, graph, train_desc, resume=False):
        saver = tf.train.Saver()
        batch_size = train_desc['batch_size']
        with tf.Session(graph=graph) as sess:
            if resume:
                saver = tf.train.import_meta_graph('saves/model.ckpt.meta')
                saver.restore(sess, tf.train.latest_checkpoint('./saves'))
                print('training resumed')
            else:
                sess.run(tf.global_variables_initializer())
            if 'pos' in train_desc:
                print('***Training POS layer***')
                for i in range(train_desc['pos']):
                    a, b, c = get_batch_pos(self, batch_size)
                    _, l = sess.run([self.optimize_op, self.loss],
                                    {self.inp: a, self.t_p: b})
                    if i % 50 == 0:
                        print(l)
                        saver.save(sess, 'saves/model.ckpt')
            if 'chunk' in train_desc:
                print('***Training chunk layer***')
                for i in range(train_desc['chunk']):
                    a, b, c = get_batch_pos(self, batch_size)
                    _, l1 = sess.run([self.optimize_op1, self.loss1], {
                        self.inp: a, self.t_p: b, self.t_c: c})
                    if i % 50 == 0:
                        print(l1)
                        saver.save(sess, 'saves/model.ckpt')
            if 'relatedness' in train_desc:
                print('***Training semantic relatedness***')
                for i in range(train_desc['relatedness']):
                    a, b, c, _ = get_batch_sent(self, batch_size)
                    _, l2 = sess.run([self.optimize_op2, self.loss2], {self.inp: a,
                                                                       self.inp1: b, self.t_rel: c})
                    if i % 50 == 0:
                        print(l2)
                        saver.save(sess, 'saves/model.ckpt')
            if 'entailment' in train_desc:
                print('***Training semantic entailment***')
                for i in range(train_desc['entailment']):
                    a, b, _, c = get_batch_sent(self, batch_size)
                    _, l3 = sess.run([self.optimize_op3, self.loss3], {self.inp: a,
                                                                       self.inp1: b, self.t_ent: c})
                    if i % 50 == 0:
                        print(l3)
                        saver.save(sess, 'saves/model.ckpt')
