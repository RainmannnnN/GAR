import numpy as np
import tensorflow as tf


def build_mlp(mlp_in, hidden_dims, act, drop_rate, is_training):
    hidden = mlp_in
    hidden = tf.layers.dense(hidden,
                             hidden_dims[0],
                             name="mlp_fc_1",
                             kernel_initializer=tf.glorot_uniform_initializer(),
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    for i in range(1, len(hidden_dims)):
        if act == 'relu':
            hidden = tf.nn.leaky_relu(hidden, alpha=0.01)  # relu 适合放在 bn 前面
        hidden = tf.layers.batch_normalization(hidden,
                                               training=is_training,
                                               name='mlp_bn_' + str(i + 1))
        if act == 'tanh':
            hidden = tf.nn.tanh(hidden)
        if drop_rate > 1e-5:
            hidden = tf.layers.dropout(hidden, rate=drop_rate, training=is_training, name='mlp_drop_' + str(i + 1))
        hidden = tf.layers.dense(hidden,
                                 hidden_dims[i],  # 最后一层不一定要是 1
                                 kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
                                 kernel_initializer=tf.glorot_uniform_initializer(),
                                 name='mlp_fc_' + str(i + 1))
    return hidden


class DSSM(object):
    def __init__(self, sess, args, emb_dim, content_dim):
        self.sess = sess
        self.emb_dim = emb_dim
        self.lys = [200, 200]
        self.act = 'tanh'
        self.dropout = 0.5
        self.d_lr = 1e-3

        self.is_training = tf.placeholder(tf.bool, name='D_is_training')
        self.target = tf.placeholder(tf.float32, shape=[None], name='target')
        self.real_emb = tf.placeholder(tf.float32, [None, emb_dim], name='real_emb')
        self.content = tf.placeholder(tf.float32, [None, content_dim], name='condition')
        self.opp_emb = tf.placeholder(tf.float32, [None, emb_dim], name='neg_emb')
        self.labels = tf.constant(np.ones(shape=[args.batch_size], dtype=np.float32))

        with tf.variable_scope("D", reuse=False):
            with tf.variable_scope("user_tower"):
                self.user_emb = build_mlp(self.opp_emb, self.lys, self.act, self.dropout, self.is_training)
            with tf.variable_scope("item_tower"):
                self.item_input = tf.concat([self.real_emb, self.content], axis=-1)
                self.item_emb = build_mlp(self.item_input, self.lys, self.act, self.dropout, self.is_training)

        if args.loss == 'BCE':
            preds = tf.reduce_sum(tf.multiply(self.user_emb, self.item_emb), axis=-1)
            self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=preds, labels=self.target))
        elif args.loss == 'BPR':
            flatten_preds = tf.reduce_sum(tf.multiply(tf.tile(self.user_emb, [2, 1]), self.item_emb), axis=-1)
            preds = tf.reshape(flatten_preds, [2, -1])
            pos_preds = tf.squeeze(tf.gather(preds, indices=[0], axis=0), axis=0)
            neg_preds = tf.squeeze(tf.gather(preds, indices=[1], axis=0), axis=0)
            diff = pos_preds - neg_preds
            self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=diff, labels=self.labels))
        self.loss += tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, scope='D'))

        # update
        self.D_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='D')
        d_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='D')
        with tf.control_dependencies(d_update_ops):
            self.d_optimizer = tf.train.AdamOptimizer(self.d_lr).minimize(self.loss, var_list=self.D_var)

        # get user rating through dot
        self.uemb = tf.placeholder(tf.float32, [None, self.emb_dim], name='user_embedding')
        self.iemb = tf.placeholder(tf.float32, [None, self.emb_dim], name='item_embedding')
        self.user_rating = tf.matmul(self.uemb, tf.transpose(self.iemb))

        # rank user rating
        self.rat = tf.placeholder(tf.float32, [None, None], name='user_rat')
        self.k = tf.placeholder(tf.int32, name='atK')
        self.top_score, self.top_item_index = tf.nn.top_k(self.rat, k=self.k)

        self.sess.run(tf.global_variables_initializer())

    def train_bce(self, content, batch_iemb, batch_uemb, batch_target):
        _, loss = self.sess.run([self.d_optimizer, self.loss],
                                feed_dict={self.opp_emb: batch_uemb,
                                           self.real_emb: batch_iemb,
                                           self.content: content,
                                           self.target: batch_target,
                                           self.is_training: True,
                                           })
        return loss

    def train_bpr(self, pos_content, pos_emb, neg_content, neg_emb, opp_emb):
        emb = np.concatenate([pos_emb, neg_emb], axis=0)
        content = np.concatenate([pos_content, neg_content], axis=0)
        _, loss = self.sess.run([self.d_optimizer, self.loss],
                                feed_dict={self.real_emb: emb,
                                           self.content: content,
                                           self.opp_emb: opp_emb,
                                           self.is_training: True,
                                           })
        return loss

    def get_user_rating(self, uids, iids, uemb, iemb):
        user_rat = self.sess.run(self.user_rating,
                                 feed_dict={self.uemb: uemb[uids],
                                            self.iemb: iemb[iids],
                                            self.is_training: False,
                                            })
        return user_rat

    def get_ranked_rating(self, ratings, k):
        ranked_score, ranked_rat = self.sess.run([self.top_score, self.top_item_index],
                                                 feed_dict={self.rat: ratings,
                                                            self.k: k})
        return ranked_score, ranked_rat

    def get_item_emb(self, content, item_emb, warm_item, cold_item):
        return self.sess.run(self.item_emb, feed_dict={self.real_emb: item_emb,
                                                       self.content: content,
                                                       self.is_training: False})

    def get_user_emb(self, user_emb):
        return self.sess.run(self.user_emb, feed_dict={self.opp_emb: user_emb,
                                                       self.is_training: False})
