import numpy as np
import tensorflow as tf


def build_mlp(mlp_in, hidden_dims, act, drop_rate, is_training, scope_name, bn=False, bn_first=False):
    with tf.variable_scope(scope_name):
        hidden = mlp_in
        if bn_first:
            hidden = tf.layers.batch_normalization(hidden,
                                                   training=is_training,
                                                   scale=False,  # 因为后面接了线性层，所以取消放缩
                                                   name='mlp_bn_1')
        hidden = tf.layers.dense(hidden,
                                 hidden_dims[0],
                                 name="mlp_fc_1",
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=1e-2),
                                 kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
        for i in range(2, len(hidden_dims) + 1):
            if act == 'relu':
                hidden = tf.nn.leaky_relu(hidden, alpha=0.01)  # relu 适合放在 bn 前面
            if bn:
                hidden = tf.layers.batch_normalization(hidden,
                                                       training=is_training,
                                                       name='mlp_bn_' + str(i))
            if act == 'tanh':
                hidden = tf.nn.tanh(hidden)
            if drop_rate > 1e-5:
                hidden = tf.layers.dropout(hidden, rate=drop_rate, training=is_training, name='mlp_drop_' + str(i))
            hidden = tf.layers.dense(hidden,
                                     hidden_dims[i - 1],  # 最后一层不一定要是 1
                                     kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
                                     kernel_initializer=tf.truncated_normal_initializer(stddev=1e-2),
                                     name='mlp_fc_' + str(i))
        return hidden


class METAEMB(object):
    def __init__(self, sess, args, emb_dim, content_dim):
        self.sess = sess
        self.emb_dim = emb_dim
        self.content_dim = content_dim
        self.g_lr = 1e-3
        self.cold_lr = self.g_lr / 10.
        self.alpha = 0.1
        self.d_drop = 0.5
        self.d_layer = [256, 256, 1]
        self.d_act = 'relu'

        self.content = tf.placeholder(tf.float32, [None, content_dim], name='condition')
        self.opp_emb = tf.placeholder(tf.float32, [None, 3, emb_dim], name='opposite_emb')
        self.target = tf.placeholder(tf.float32, [None, 1], name='label')
        self.g_training = tf.placeholder(tf.bool, name='generator_training_flag')

        with tf.variable_scope('G'):
            gen_emb_0 = build_mlp(self.content, [200, 200], 'tanh', 0.0, self.g_training, 'E0')
            gen_emb_1 = build_mlp(self.content, [200, 200], 'tanh', 0.0, self.g_training, 'E1')
            gen_emb_2 = build_mlp(self.content, [200, 200], 'tanh', 0.0, self.g_training, 'E2')
            self.gen_emb = tf.stack([gen_emb_0, gen_emb_1, gen_emb_2], axis=1)

        gen_output_a = self.build_discriminator(self.opp_emb, self.gen_emb,
                                                self.d_layer, self.d_act, self.d_drop,
                                                training=False, reuse=False, rating=False)
        self.loss_a = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=gen_output_a, labels=self.target))
        gen_emb_grads = tf.gradients(self.loss_a, self.gen_emb)[0]
        gen_emb_new = self.gen_emb - self.cold_lr * gen_emb_grads
        gen_output_b = self.build_discriminator(self.opp_emb, gen_emb_new,
                                                self.d_layer, self.d_act, self.d_drop,
                                                training=False, reuse=True, rating=False)
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

        # get user rating
        self.uemb = tf.placeholder(tf.float32, [None, 3, self.emb_dim], name='user_embedding')
        self.iemb = tf.placeholder(tf.float32, [None, 3, self.emb_dim], name='item_embedding')
        self.user_rating = self.build_discriminator(self.uemb, self.iemb,
                                                    self.d_layer, self.d_act, self.d_drop,
                                                    training=False, reuse=True, rating=True)
        self.user_rating = tf.reshape(self.user_rating, [tf.shape(self.uemb)[0], -1])

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
                       self.target: lbs.reshape([-1, 1]),
                       self.g_training: True})
        return g_loss

    def build_discriminator(self, uembs, iembs, hid_dims, act, drop_rate, training, reuse, rating):
        with tf.variable_scope("D", reuse=reuse):
            u_bn_out = []
            i_bn_out = []
            for i in range(3):
                current_u_in = tf.squeeze(tf.gather(uembs, indices=[i], axis=1), axis=1)
                u_bn_out.append(
                    tf.layers.batch_normalization(current_u_in,
                                                  training=training,
                                                  name="u_input_bn_" + str(i),
                                                  reuse=reuse))
                current_i_in = tf.squeeze(tf.gather(iembs, indices=[i], axis=1), axis=1)
                i_bn_out.append(
                    tf.layers.batch_normalization(current_i_in,
                                                  training=training,
                                                  name="i_input_bn_" + str(i),
                                                  reuse=reuse))
            uemb_bn = tf.stack(u_bn_out, axis=1)  # (batch, 3, emb)
            iemb_bn = tf.stack(i_bn_out, axis=1)

            if not rating:
                # train
                train_p_list = [tf.reduce_sum(uemb_bn * tf.roll(iemb_bn, shift=_, axis=1), axis=2) for _ in range(3)]
                train_mlp_in = tf.concat(train_p_list, axis=1)
                mlp_out = build_mlp(train_mlp_in, hid_dims, act, drop_rate,
                                    training, scope_name='mlp', bn=True, bn_first=True)
            else:
                rating_p_list = [tf.matmul(
                    tf.transpose(uemb_bn, [1, 0, 2]),
                    tf.transpose(tf.roll(iemb_bn, shift=_, axis=1), [1, 2, 0])) for _ in range(3)]
                rating_mlp_in = tf.reshape(tf.transpose(tf.concat(rating_p_list, axis=0), [1, 2, 0]), [-1, 9])
                mlp_out = build_mlp(rating_mlp_in, hid_dims, act, drop_rate,
                                    training, scope_name='mlp', bn=True, bn_first=True)
        return mlp_out

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
