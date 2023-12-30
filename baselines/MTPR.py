import tensorflow as tf
import numpy as np


def build_mlp(mlp_in, is_training, scope, drop=0.0, reuse=False):
    with tf.variable_scope(scope, reuse=reuse):
        out = tf.layers.dense(mlp_in, 200,
                              kernel_initializer=tf.glorot_uniform_initializer(),
                              kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
                              )
        # hidden = tf.nn.tanh(hidden)
        # out = tf.layers.dense(hidden, 200,
        #                       kernel_initializer=tf.glorot_uniform_initializer(),
        #                       kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
        #                       )
    return out


class MTPR(object):
    """
    Note: 这里的 embeddings 是原始的 node2vec 模型的输出。
    """

    def __init__(self, sess, args, emb_dim, content_dim):
        self.sess = sess
        self.emb_dim = emb_dim
        self.content_dim = content_dim
        self.g_lr = 1e-3
        self.drop = 0.0

        self.pos_content = tf.placeholder(tf.float32, [None, content_dim], name='positive_content')
        self.pos_emb = tf.placeholder(tf.float32, [None, emb_dim], name='positive_embeddings')
        self.neg_content = tf.placeholder(tf.float32, [None, content_dim], name='negative_content')
        self.neg_emb = tf.placeholder(tf.float32, [None, emb_dim], name='negative_embeddings')

        self.opp_emb = tf.placeholder(tf.float32, [None, emb_dim], name='opposite_emb')
        self.g_training = tf.placeholder(tf.bool, name='generator_training_flag')
        self.masked_real_emb = tf.constant(np.zeros(shape=[args.batch_size, emb_dim], dtype=np.float32))
        self.ones_label = tf.constant(np.ones(shape=[args.batch_size], dtype=np.float32))

        # build Student - generator
        with tf.variable_scope('G'):
            self.user_emb = build_mlp(self.opp_emb, self.g_training, 'user_mlp', self.drop, False)

            # 表现差可能是因为分成了四个批次输入 item MLP 中，而 item MLP 中有 BatchNorm 层，分批输入导致归一化操作不一致
            self.pos_item_with_emb = build_mlp(
                tf.concat([self.pos_emb, self.pos_content], axis=1), self.g_training, 'item_mlp', self.drop, False)
            self.pos_item_without_emb = build_mlp(
                tf.concat([self.masked_real_emb, self.pos_content], axis=1), self.g_training, 'item_mlp', self.drop,
                True)

            self.neg_item_with_emb = build_mlp(
                tf.concat([self.neg_emb, self.neg_content], axis=1), self.g_training, 'item_mlp', self.drop, True)
            self.neg_item_without_emb = build_mlp(
                tf.concat([self.masked_real_emb, self.neg_content], axis=1), self.g_training, 'item_mlp', self.drop,
                True)

        # 4 kinds of outputs
        self.pos_out_with_emb = tf.reduce_sum(tf.multiply(self.user_emb, self.pos_item_with_emb), axis=-1)
        self.pos_out_without_emb = tf.reduce_sum(tf.multiply(self.user_emb, self.pos_item_without_emb), axis=-1)
        self.neg_out_with_emb = tf.reduce_sum(tf.multiply(self.user_emb, self.neg_item_with_emb), axis=-1)
        self.neg_out_without_emb = tf.reduce_sum(tf.multiply(self.user_emb, self.neg_item_without_emb), axis=-1)

        """Construct Loss"""
        # homogenous
        self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=self.pos_out_with_emb - self.neg_out_with_emb, labels=self.ones_label))
        self.g_loss += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=self.pos_out_without_emb - self.neg_out_without_emb, labels=self.ones_label))
        # heterogenous
        self.g_loss += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=self.pos_out_with_emb - self.neg_out_without_emb, labels=self.ones_label))
        self.g_loss += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=self.pos_out_without_emb - self.neg_out_with_emb, labels=self.ones_label))

        # regularization
        # self.g_reg_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, scope='G')
        # self.g_loss = tf.add_n([self.g_loss] + self.g_reg_loss)

        self.G_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='G')
        g_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='G')
        with tf.control_dependencies(g_update_ops):
            self.g_train_step = tf.train.AdamOptimizer(self.g_lr).minimize(
                self.g_loss, var_list=self.G_var)

        # get user rating through dot
        self.uemb = tf.placeholder(tf.float32, [None, self.emb_dim], name='user_embedding')
        self.iemb = tf.placeholder(tf.float32, [None, self.emb_dim], name='item_embedding')
        self.user_rating = tf.matmul(self.uemb, tf.transpose(self.iemb))

        # rank user rating
        self.rat = tf.placeholder(tf.float32, [None, None], name='user_rat')
        self.k = tf.placeholder(tf.int32, name='atK')
        self.top_score, self.top_item_index = tf.nn.top_k(self.rat, k=self.k)

        # get cold item representations
        with tf.variable_scope('G'):
            self.item_without_emb = build_mlp(
                tf.concat([tf.zeros(shape=[tf.shape(self.pos_content)[0], self.emb_dim]), self.pos_content], axis=1),
                self.g_training, 'item_mlp', self.drop, True)

        self.sess.run(tf.global_variables_initializer())

    def train_bpr(self, pos_content, pos_emb, neg_content, neg_emb, opp_emb):
        _, g_loss, = self.sess.run(
            [self.g_train_step, self.g_loss],
            feed_dict={self.pos_content: pos_content,
                       self.pos_emb: pos_emb,
                       self.neg_content: neg_content,
                       self.neg_emb: neg_emb,
                       self.opp_emb: opp_emb,
                       self.g_training: True,
                       })
        return g_loss

    def train_mapping(self, batch_uemb, batch_iemb):
        return None

    def get_user_rating(self, uids, iids, uemb, iemb):
        user_rat = self.sess.run(self.user_rating,
                                 feed_dict={self.uemb: uemb[uids],
                                            self.iemb: iemb[iids], })
        return user_rat

    def get_ranked_rating(self, ratings, k):
        ranked_score, ranked_rat = self.sess.run([self.top_score, self.top_item_index],
                                                 feed_dict={self.rat: ratings,
                                                            self.k: k})
        return ranked_score, ranked_rat

    # get generated embeddings
    def get_item_emb(self, content, item_emb, warm_item, cold_item):
        out_emb = np.copy(item_emb)
        out_emb[warm_item] = self.sess.run(self.pos_item_with_emb, feed_dict={self.pos_content: content[warm_item],
                                                                              self.pos_emb: item_emb[warm_item],
                                                                              self.g_training: False})
        out_emb[cold_item] = self.sess.run(self.item_without_emb, feed_dict={self.pos_content: content[cold_item],
                                                                             self.g_training: False})

        return out_emb

    def get_user_emb(self, user_emb):
        return self.sess.run(self.user_emb, feed_dict={self.opp_emb: user_emb,
                                                       self.g_training: False})
