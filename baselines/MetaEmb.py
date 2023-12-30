import numpy as np
import tensorflow as tf


def build_mlp(mlp_in, is_training):
    hidden = tf.layers.dense(mlp_in, 200,
                             kernel_initializer=tf.truncated_normal_initializer(stddev=1e-2),
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-4),
                             name='dense_1',
                             )
    hidden = tf.nn.tanh(hidden)
    out = tf.layers.dense(hidden, 200,
                          kernel_initializer=tf.truncated_normal_initializer(stddev=1e-2),
                          kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-4),
                          name='dense_2',
                          )
    return out


class METAEMB(object):
    def __init__(self, sess, args, emb_dim, content_dim):
        self.sess = sess
        self.emb_dim = emb_dim
        self.content_dim = content_dim
        self.g_lr = 1e-3
        self.cold_lr = self.g_lr / 10.
        self.alpha = 0.1

        self.content = tf.placeholder(tf.float32, [None, content_dim], name='condition')
        self.opp_emb = tf.placeholder(tf.float32, [None, emb_dim], name='opposite_emb')
        self.target = tf.placeholder(tf.float32, [None], name='label')
        self.g_training = tf.placeholder(tf.bool, name='generator_training_flag')

        # build generator and discriminator's output
        with tf.variable_scope('G'):
            self.gen_emb = build_mlp(self.content, self.g_training)
            gen_output_a = tf.reduce_sum(tf.multiply(self.opp_emb, self.gen_emb), axis=1)
            self.loss_a = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=gen_output_a, labels=self.target))

            gen_emb_grads = tf.gradients(self.loss_a, self.gen_emb)[0]
            gen_emb_new = self.gen_emb - self.cold_lr * gen_emb_grads

            gen_output_b = tf.reduce_sum(tf.multiply(self.opp_emb, gen_emb_new), axis=1)
            self.loss_b = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=gen_output_b, labels=self.target))

            self.g_loss = self.alpha * self.loss_a + (1 - self.alpha) * self.loss_b

        self.g_reg_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, scope='G')
        self.g_loss = tf.add_n([self.g_loss] + self.g_reg_loss)

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

        self.sess.run(tf.global_variables_initializer())

    def train_bce(self, content, real_emb, opp_emb, lbs):
        _, g_loss, = self.sess.run(
            [self.g_train_step, self.g_loss],
            feed_dict={self.content: content,
                       self.opp_emb: opp_emb,
                       self.target: lbs,
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

    def get_item_emb(self, content, item_emb, warm_item, cold_item):
        out_emb = np.copy(item_emb)
        out_emb[cold_item] = self.sess.run(self.gen_emb, feed_dict={self.content: content[cold_item],
                                                                    self.g_training: False})
        return out_emb

    def get_user_emb(self, user_emb):
        return user_emb
