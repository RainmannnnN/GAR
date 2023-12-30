import numpy as np
import tensorflow as tf
from pprint import pprint


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


class METAEMB(object):
    def __init__(self, sess, args, emb_dim, content_dim):
        self.sess = sess
        self.emb_dim = emb_dim
        self.content_dim = content_dim
        self.act = 'tanh'
        self.lys = [200, 200]
        self.dropout = 0.5
        self.g_lr = 1e-3
        self.cold_lr = self.g_lr / 10.
        self.alpha = 0.1

        self.real_emb = tf.placeholder(tf.float32, [None, emb_dim], name='real_emb')
        self.content = tf.placeholder(tf.float32, [None, content_dim], name='condition')
        self.opp_emb = tf.placeholder(tf.float32, [None, emb_dim], name='opposite_emb')
        self.target = tf.placeholder(tf.float32, [None], name='label')
        self.g_training = tf.placeholder(tf.bool, name='generator_training_flag')
        self.labels = tf.constant(np.ones(shape=[args.batch_size], dtype=np.float32))

        # build user/item emb
        with tf.variable_scope('G', reuse=False):
            gen_item_id_emb = build_mlp(self.content, self.lys, self.act, 0.0, self.g_training)
        self.drop_indicator = tf.placeholder(tf.float32, shape=[None, 1], name='drop_i_indicator')
        with tf.variable_scope("D", reuse=False):
            with tf.variable_scope("user_tower"):
                self.user_emb = build_mlp(self.opp_emb, self.lys, self.act, self.dropout, self.g_training)
            with tf.variable_scope("item_tower"):
                self.item_input = tf.concat(
                    [self.real_emb * self.drop_indicator + gen_item_id_emb * (1 - self.drop_indicator),
                     self.content], axis=-1)
                self.item_emb = build_mlp(self.item_input, self.lys, self.act, self.dropout, self.g_training)

        # 
        with tf.variable_scope("D", reuse=True):
            with tf.variable_scope('item_tower', reuse=True):
                gen_item_input = tf.concat([gen_item_id_emb, self.content], axis=-1)
                gen_item_emb = build_mlp(gen_item_input, self.lys, self.act, self.dropout, self.g_training)
                flatten_preds = tf.reduce_sum(tf.multiply(tf.tile(self.user_emb, [2, 1]), gen_item_emb), axis=-1)
                preds = tf.reshape(flatten_preds, [2, -1])
                pos_preds = tf.squeeze(tf.gather(preds, indices=[0], axis=0), axis=0)
                neg_preds = tf.squeeze(tf.gather(preds, indices=[1], axis=0), axis=0)
                diff = pos_preds - neg_preds
                loss_a = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=diff, labels=self.labels))

                # gen_output_a = tf.reduce_sum(tf.multiply(self.opp_emb, gen_item_emb), axis=1)
                # loss_a = tf.reduce_mean(
                #     tf.nn.sigmoid_cross_entropy_with_logits(logits=gen_output_a, labels=self.target))
                
                gen_emb_grads = tf.gradients(loss_a, gen_item_id_emb)[0]
                gen_item_id_emb_new = gen_item_id_emb - self.cold_lr * gen_emb_grads
                gen_item_input_new = tf.concat([gen_item_id_emb_new, self.content], axis=-1)
                gen_item_emb_new = build_mlp(gen_item_input_new, self.lys, self.act, self.dropout, self.g_training)
                flatten_preds_new = tf.reduce_sum(tf.multiply(tf.tile(self.user_emb, [2, 1]), gen_item_emb_new), axis=-1)
                preds_new = tf.reshape(flatten_preds_new, [2, -1])
                pos_preds_new = tf.squeeze(tf.gather(preds_new, indices=[0], axis=0), axis=0)
                neg_preds_new = tf.squeeze(tf.gather(preds_new, indices=[1], axis=0), axis=0)
                diff_new = pos_preds_new - neg_preds_new
                loss_b = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=diff_new, labels=self.labels))

                # gen_output_b = tf.reduce_sum(tf.multiply(self.opp_emb, gen_item_emb_new), axis=1)
                # loss_b = tf.reduce_mean(
                #     tf.nn.sigmoid_cross_entropy_with_logits(logits=gen_output_b, labels=self.target))

        self.g_loss = self.alpha * loss_a + (1 - self.alpha) * loss_b
        self.g_reg_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, scope='G')
        self.g_loss = tf.add_n([self.g_loss] + self.g_reg_loss)
        self.G_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='G')
        print("trainble variables:")
        pprint(self.G_var)
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

    def train_bpr(self, pos_content, pos_emb, neg_content, neg_emb, opp_emb):
        emb = np.concatenate([pos_emb, neg_emb], axis=0)
        content = np.concatenate([pos_content, neg_content], axis=0)
        _, g_loss, = self.sess.run(
            [self.g_train_step, self.g_loss],
            feed_dict={self.content: content,
                       self.opp_emb: opp_emb,
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
        drop_indicator = np.ones([len(item_emb), 1])
        drop_indicator[cold_item] = 0
        ret_item_emb = self.sess.run(self.item_emb, feed_dict={self.real_emb: item_emb,
                                                               self.content: content,
                                                               self.drop_indicator: drop_indicator,
                                                               self.g_training: False})
        return ret_item_emb

    def get_user_emb(self, user_emb):
        ret_user_emb = self.sess.run(self.user_emb, feed_dict={self.opp_emb: user_emb,
                                                               self.g_training: False})
        return ret_user_emb
