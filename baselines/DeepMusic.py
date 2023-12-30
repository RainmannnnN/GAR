import numpy as np
import tensorflow as tf


def build_mlp(mlp_in, is_training):
    hidden = tf.layers.dense(mlp_in, 200,
                             kernel_initializer=tf.truncated_normal_initializer(stddev=1e-2),
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-4),
                             )
    hidden = tf.nn.tanh(hidden)
    out = tf.layers.dense(hidden, 200,  # 最后一层不一定要是 1
                          kernel_initializer=tf.truncated_normal_initializer(stddev=1e-2),
                          kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-4),
                          )
    return out


class DEEPMUSIC(object):
    def __init__(self, sess, args, emb_dim, content_dim):
        self.sess = sess
        self.emb_dim = emb_dim
        self.content_dim = content_dim

        self.d_lr = 1e-3

        self.content = tf.placeholder(tf.float32, [None, content_dim], name='condition')
        self.real_emb = tf.placeholder(tf.float32, [None, emb_dim], name='real_emb')
        self.g_training = tf.placeholder(tf.bool, name='generator_training_flag')

        # build generator and discriminator's output
        with tf.variable_scope('G'):
            self.gen_emb = build_mlp(self.content, self.g_training)

        self.g_loss = tf.reduce_mean(tf.reduce_sum(tf.abs(self.gen_emb - self.real_emb), axis=-1))
        self.g_reg_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, scope='G')
        self.g_loss = tf.add_n([self.g_loss] + self.g_reg_loss)

        self.G_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='G')
        g_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='G')
        with tf.control_dependencies(g_update_ops):
            self.g_train_step = tf.train.AdamOptimizer(self.d_lr).minimize(
                self.g_loss, var_list=self.G_var)

        # get user rating through dot
        self.uemb = tf.placeholder(tf.float32, [None, self.emb_dim], name='user_embedding')
        self.iemb = tf.placeholder(tf.float32, [None, self.emb_dim], name='item_embedding')
        self.user_rating = tf.matmul(self.uemb, tf.transpose(self.iemb))

        # rank user rating
        self.rat = tf.placeholder(tf.float32, [None, None], name='user_rat')
        self.k = tf.placeholder(tf.int32, name='atK')
        self.top_score, self.top_item_index = tf.nn.top_k(self.rat, k=self.k)

        self.sess.run(tf.global_variables_initializer())

    def train_sim(self, content, real_emb):
        _, g_loss, = self.sess.run([self.g_train_step, self.g_loss],
                                   feed_dict={self.content: content,
                                              self.real_emb: real_emb,
                                              self.g_training: True})
        return g_loss

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
        out_emb[cold_item] = self.sess.run(self.gen_emb, feed_dict={self.content: content[cold_item],
                                                                    self.g_training: False})
        return out_emb

    def get_user_emb(self, user_emb):
        return user_emb


# class LAB_SIM_MLP(object):
#     def __init__(self, sess, args, emb_dim, content_dim):
#         self.sess = sess
#         self.emb_dim = emb_dim
#         self.content_dim = content_dim
#         self.g_lr = 1e-3
#
#         self.content = tf.placeholder(tf.float32, [None, content_dim], name='condition')
#         self.real_emb = tf.placeholder(tf.float32, [None, emb_dim], name='real_emb')
#         self.opp_emb = tf.placeholder(tf.float32, [None, emb_dim], name='opposite_emb')
#         self.g_training = tf.placeholder(tf.bool, name='generator_training_flag')
#
#         # build generator and discriminator's output
#         with tf.variable_scope('G'):
#             self.gen_emb = build_mlp(self.content, self.g_training)
#         self.real_output = tf.reduce_sum(tf.multiply(self.real_emb, self.opp_emb), axis=-1)
#         self.gen_output = tf.reduce_sum(tf.multiply(self.gen_emb, self.opp_emb), axis=-1)
#
#         # construct loss
#         self.g_loss = tf.reduce_mean(tf.abs(self.real_output - self.gen_output))
#         self.g_reg_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, scope='G')
#         self.g_loss = tf.add_n([self.g_loss] + self.g_reg_loss)
#
#         self.G_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='G')
#         g_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='G')
#         with tf.control_dependencies(g_update_ops):
#             self.g_train_step = tf.train.AdamOptimizer(self.g_lr).minimize(
#                 self.g_loss, var_list=self.G_var)
#
#         # get user rating through dot
#         self.uemb = tf.placeholder(tf.float32, [None, self.emb_dim], name='user_embedding')
#         self.iemb = tf.placeholder(tf.float32, [None, self.emb_dim], name='item_embedding')
#         self.user_rating = tf.matmul(self.uemb, tf.transpose(self.iemb))
#
#         # rank user rating
#         self.rat = tf.placeholder(tf.float32, [None, None], name='user_rat')
#         self.k = tf.placeholder(tf.int32, name='atK')
#         self.top_score, self.top_item_index = tf.nn.top_k(self.rat, k=self.k)
#
#         self.sess.run(tf.global_variables_initializer())
#
#     def train_bce(self, content, real_emb, opp_emb, lbs):
#         _, g_loss, = self.sess.run(
#             [self.g_train_step, self.g_loss],
#             feed_dict={self.content: content,
#                        self.real_emb: real_emb,
#                        self.opp_emb: opp_emb,
#                        self.g_training: True})
#         return g_loss
#
#     def train_mapping(self, batch_uemb, batch_iemb):
#         return None
#
#     def get_user_rating(self, uids, iids, uemb, iemb):
#         user_rat = self.sess.run(self.user_rating,
#                                  feed_dict={self.uemb: uemb[uids],
#                                             self.iemb: iemb[iids], })
#         return user_rat
#
#     def get_ranked_rating(self, ratings, k):
#         ranked_score, ranked_rat = self.sess.run([self.top_score, self.top_item_index],
#                                    feed_dict={self.rat: ratings,
#                                               self.k: k})
#         return ranked_score, ranked_rat
#
#     # get generated embeddings
#     def get_item_emb(self, content, item_emb, warm_item, cold_item):
#         out_emb = np.copy(item_emb)
#         out_emb[cold_item] = self.sess.run(self.gen_emb, feed_dict={self.content: content[cold_item],
#                                                                     self.g_training: False})
#         return out_emb
#
#     def get_user_emb(self, user_emb):
#         return user_emb
