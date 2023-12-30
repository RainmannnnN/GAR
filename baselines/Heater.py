import tensorflow as tf
import numpy as np


def l2_norm(para):
    return tf.reduce_sum(tf.square(para))

def dense_batch_fc_tanh(x, units, is_training, scope, do_norm=False):
    with tf.variable_scope(scope):
        init = tf.truncated_normal_initializer(stddev=0.01)
        h1_w = tf.get_variable(scope + '_w', shape=[x.get_shape().as_list()[1], units], initializer=init)
        h1_b = tf.get_variable(scope + '_b', shape=[1, units], initializer=tf.zeros_initializer())
        h1 = tf.matmul(x, h1_w) + h1_b
        if do_norm:
            h2 = tf.layers.batch_normalization(h1, training=is_training, name=scope + '_bn')
            return tf.nn.tanh(h2), l2_norm(h1_w) + l2_norm(h1_b)
        else:
            return tf.nn.tanh(h1), l2_norm(h1_w) + l2_norm(h1_b)


def dense_fc(x, units, scope):
    with tf.variable_scope(scope):
        init = tf.truncated_normal_initializer(stddev=0.01)
        h1_w = tf.get_variable(scope + '_w', shape=[x.get_shape().as_list()[1], units], initializer=init)
        h1_b = tf.get_variable(scope + '_b', shape=[1, units], initializer=tf.zeros_initializer())
        h1 = tf.matmul(x, h1_w) + h1_b
        return h1, l2_norm(h1_w) + l2_norm(h1_b)


class HEATER:
    def __init__(self, sess, args, emb_dim, content_dim):
        self.sess = sess
        self.emb_dim = emb_dim  # input embedding dimension
        self.transform_layers = [200, 200]  # output dimension
        self.content_dim = content_dim
        self.expert_layers = [200]
        self.n_experts = args.n_experts  # num of experts
        self.reg = args.reg  # coefficient of regularization
        self.alpha = args.alpha  # coefficient of diff loss
        self.reg_loss = 0.
        self.lr = 1e-3

        self.is_training = tf.placeholder(tf.bool, name='is_training')
        self.target = tf.placeholder(tf.float32, shape=[None], name='target')
        self.Uin = tf.placeholder(tf.float32, shape=[None, self.emb_dim], name='u_emb')
        self.Vin = tf.placeholder(tf.float32, shape=[None, self.emb_dim], name='v_emb')

        # item
        self.Vcontent = tf.placeholder(tf.float32, shape=[None, self.content_dim])
        self.dropout_item_indicator = tf.placeholder(tf.float32, [None, 1], name='dropout_flag')

        vcontent_gate, vcontent_gate_reg = dense_fc(self.Vcontent, self.n_experts, 'vcontent_gate_layer')
        vcontent_gate = tf.nn.tanh(vcontent_gate)
        self.reg_loss += vcontent_gate_reg
        vcontent_expert_list = []
        for i in range(self.n_experts):
            tmp_expert = self.Vcontent
            for ihid, hid in enumerate(self.expert_layers):
                tmp_expert, tmp_reg = dense_fc(tmp_expert, hid, 'Vexpert_' + str(ihid) + '_' + str(i))
                tmp_expert = tf.nn.tanh(tmp_expert)
                self.reg_loss += tmp_reg
            vcontent_expert_list.append(tf.expand_dims(tmp_expert, axis=1))
        vcontent_expert_concat = tf.concat(vcontent_expert_list, axis=1)
        Vcontent_last = tf.matmul(tf.expand_dims(vcontent_gate, axis=1), vcontent_expert_concat)
        Vcontent_last = tf.squeeze(tf.nn.tanh(Vcontent_last), axis=1)

        diff_item_loss = tf.reduce_mean(tf.reduce_sum(tf.square(Vcontent_last - self.Vin), axis=1))
        v_last = (self.Vin * self.dropout_item_indicator + Vcontent_last * (1 - self.dropout_item_indicator))

        # user
        u_last = self.Uin
        diff_user_loss = 0

        # calc recommendation loss
        for ihid, hid in enumerate(self.transform_layers[:-1]):
            u_last, u_reg = dense_batch_fc_tanh(u_last, hid, self.is_training, 'user_layer_%d' % ihid, do_norm=True)
            v_last, v_reg = dense_batch_fc_tanh(v_last, hid, self.is_training, 'item_layer_%d' % ihid, do_norm=True)
            self.reg_loss += u_reg
            self.reg_loss += v_reg

        u_last, u_reg = dense_fc(u_last, self.transform_layers[-1], 'user_output')
        self.reg_loss += u_reg
        v_last, v_reg = dense_fc(v_last, self.transform_layers[-1], 'item_output')
        self.reg_loss += v_reg

        self.U_embedding = u_last
        self.V_embedding = v_last

        with tf.variable_scope("loss"):
            self.diff_loss = diff_item_loss + diff_user_loss
            self.preds = tf.reduce_sum(tf.multiply(self.U_embedding, self.V_embedding), axis=-1)
            # self.target_loss = tf.reduce_mean(tf.squared_difference(self.preds, self.target))
            self.target_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.preds, labels=self.target))
            self.loss = self.target_loss + self.reg * self.reg_loss + self.alpha * self.diff_loss

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

        # get user rating through dot
        self.uemb = tf.placeholder(tf.float32, [None, self.emb_dim], name='user_embedding')
        self.iemb = tf.placeholder(tf.float32, [None, self.emb_dim], name='item_embedding')
        self.user_rating = tf.matmul(self.uemb, tf.transpose(self.iemb))

        # rank user rating
        self.rat = tf.placeholder(tf.float32, [None, None], name='user_rat')
        self.k = tf.placeholder(tf.int32, name='atK')
        self.top_score, self.top_item_index = tf.nn.top_k(self.rat, k=self.k)

        self.sess.run(tf.global_variables_initializer())

    def train_bce(self, v_content, v_emb, u_emb, target):
        drop_indicator = np.random.randint(2, size=(len(v_emb), 1)).astype(np.float32)
        _train_dict = {
            self.is_training: True,
            self.Vcontent: v_content,
            self.Uin: u_emb,
            self.Vin: v_emb,
            self.target: target,
            self.dropout_item_indicator: drop_indicator
        }
        _, loss = self.sess.run([self.optimizer, self.loss], feed_dict=_train_dict)
        return loss

    def get_user_rating(self, u_array, v_array, u_emb, v_emb):
        rating = self.sess.run(self.user_rating, feed_dict={self.is_training: False,
                                                            self.uemb: u_emb[u_array],
                                                            self.iemb: v_emb[v_array], })
        return rating

    def get_item_emb(self, content, item_emb, warm_item, cold_item):
        drop_indicator = np.ones((len(content), 1), dtype=np.float32)
        drop_indicator[cold_item] = 0

        return self.sess.run(self.V_embedding, feed_dict={self.Vcontent: content,
                                                          self.Vin: item_emb,
                                                          self.dropout_item_indicator: drop_indicator,
                                                          self.is_training: False})

    def get_user_emb(self, user_emb):
        return self.sess.run(self.U_embedding, feed_dict={self.Uin: user_emb,
                                                          self.is_training: False})

    def get_ranked_rating(self, ratings, k):
        ranked_score, ranked_rat = self.sess.run([self.top_score, self.top_item_index],
                                   feed_dict={self.rat: ratings,
                                              self.k: k})
        return ranked_score, ranked_rat
