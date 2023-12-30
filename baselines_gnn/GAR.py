import numpy as np
import tensorflow as tf


def build_mlp(mlp_in, hidden_dims, act, drop_rate, is_training, scope_name, bn_first=True):
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
                                 # kernel_initializer=tf.truncated_normal_initializer(0.02),
                                 kernel_initializer=tf.glorot_uniform_initializer(),
                                 kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
        for i in range(2, len(hidden_dims) + 1):
            if act == 'relu':
                hidden = tf.nn.leaky_relu(hidden, alpha=0.01)  # relu 适合放在 bn 前面
            hidden = tf.layers.batch_normalization(hidden,
                                                   training=is_training,
                                                   name='mlp_bn_' + str(i))
            if act == 'tanh':
                hidden = tf.nn.tanh(hidden)
            hidden = tf.layers.dropout(hidden, rate=drop_rate, training=is_training, name='mlp_drop_' + str(i))
            hidden = tf.layers.dense(hidden,
                                     hidden_dims[i - 1],  # 最后一层不一定要是 1
                                     # kernel_initializer=tf.truncated_normal_initializer(0.02),
                                     kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
                                     kernel_initializer=tf.glorot_uniform_initializer(),
                                     name='mlp_fc_' + str(i))
        return hidden


class GAR(object):
    def __init__(self, sess, args, emb_dim, content_dim):
        self.sess = sess
        self.emb_dim = emb_dim
        self.content_dim = content_dim
        self.g_lr = 1e-3
        self.d_lr = 1e-3
        self.g_drop = 0.1
        self.d_drop = 0.5
        self.g_layer = [200, 200]
        self.d_layer = [256, 256, 1]
        self.g_act = 'tanh'
        self.d_act = 'relu'

        self.content = tf.placeholder(tf.float32, [None, content_dim], name='condition')
        self.real_emb = tf.placeholder(tf.float32, [None, 3, emb_dim], name='real_emb')
        self.neg_emb = tf.placeholder(tf.float32, [None, 3, emb_dim], name='neg_emb')
        self.opp_emb = tf.placeholder(tf.float32, [None, 3, emb_dim], name='opp_emb')
        self.g_training = tf.placeholder(tf.bool, name='G_is_training')
        self.d_training = tf.placeholder(tf.bool, name='D_is_training')

        # build generator and discriminator's output
        self.gen_emb = self.build_generator(
            self.content, self.g_layer, self.g_act, self.g_drop, self.g_training, False
        )

        # D loss
        # 重要：必须将 uemb 和 iemb 同批输入，分批（pos，neg）不能正常训练，可能是 bn 的缘故
        uemb = tf.tile(self.opp_emb, [3, 1, 1])
        iemb = tf.concat([self.real_emb, self.neg_emb, self.gen_emb], axis=0)
        D_out = self.build_discriminator(uemb, iemb,
                                         self.d_layer, self.d_act, self.d_drop, self.d_training, False, False)

        self.D_out = tf.transpose(tf.reshape(D_out, [3, -1]))
        self.real_logit = tf.gather(self.D_out, indices=[0], axis=1)
        self.neg_logit = tf.gather(self.D_out, indices=[1], axis=1)
        self.d_fake_logit = tf.gather(self.D_out, indices=[2], axis=1)

        self.d_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=self.real_logit - args.alpha * self.neg_logit - (1 - args.alpha) * self.d_fake_logit,
            labels=tf.ones_like(self.real_logit)))
        self.d_loss += tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, scope='D'))

        # G loss
        self.g_out = self.build_discriminator(self.opp_emb, self.gen_emb,
                                              self.d_layer, self.d_act, self.d_drop,
                                              self.d_training, True, False)
        self.d_out = self.build_discriminator(self.opp_emb, self.real_emb,
                                              self.d_layer, self.d_act, self.d_drop,
                                              self.d_training, True, False)
        self.g_loss = (1.0 - args.sim_coe) * tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=self.g_out - self.d_out, labels=tf.ones_like(self.g_out)))
        # self.sim_loss = args.sim_coe * tf.reduce_mean(tf.abs(self.gen_emb - self.real_emb))  # emb similarity
        self.sim_loss = args.sim_coe * (tf.reduce_mean(tf.reduce_sum(tf.abs(self.gen_emb - self.real_emb), axis=-1)))
        self.g_loss += self.sim_loss
        self.g_loss += tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, scope='G'))

        # update
        d_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='D')
        with tf.control_dependencies(d_update_ops):
            self.d_optimizer = tf.train.AdamOptimizer(self.d_lr).minimize(self.d_loss,
                                                                          var_list=tf.get_collection(
                                                                              tf.GraphKeys.TRAINABLE_VARIABLES,
                                                                              scope='D'))
        g_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='G')
        with tf.control_dependencies(g_update_ops):
            self.g_optimizer = tf.train.AdamOptimizer(self.g_lr).minimize(self.g_loss,
                                                                          var_list=tf.get_collection(
                                                                              tf.GraphKeys.TRAINABLE_VARIABLES,
                                                                              scope='G'))

        # get user rating through dot
        self.uemb = tf.placeholder(tf.float32, [None, 3, self.emb_dim], name='user_embedding')
        self.iemb = tf.placeholder(tf.float32, [None, 3, self.emb_dim], name='item_embedding')
        self.user_rating = self.build_discriminator(self.uemb, self.iemb,
                                                    self.d_layer, self.d_act, self.d_drop,
                                                    self.d_training, True, True)
        self.user_rating = tf.reshape(self.user_rating, [tf.shape(self.uemb)[0], -1])

        # rank user rating
        self.rat = tf.placeholder(tf.float32, [None, None], name='user_rat')
        self.k = tf.placeholder(tf.int32, name='atK')
        self.top_score, self.top_item_index = tf.nn.top_k(self.rat, k=self.k)

        self.sess.run(tf.global_variables_initializer())

    def build_generator(self, condition, hid_dims, act, drop_rate, training, reuse):
        with tf.variable_scope('G', reuse=reuse):
            gen_emb_0 = build_mlp(condition, hid_dims, act, drop_rate, training, 'E0', False)
            gen_emb_1 = build_mlp(condition, hid_dims, act, drop_rate, training, 'E1', False)
            gen_emb_2 = build_mlp(condition, hid_dims, act, drop_rate, training, 'E2', False)
        return tf.stack([gen_emb_0, gen_emb_1, gen_emb_2], axis=1)

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
                                    training, scope_name='mlp', bn_first=True)
            else:
                rating_p_list = [tf.matmul(tf.transpose(uemb_bn, [1, 0, 2]),
                                           tf.transpose(tf.roll(iemb_bn, shift=_, axis=1), [1, 2, 0])) for _ in
                                 range(3)]
                rating_mlp_in = tf.reshape(tf.transpose(tf.concat(rating_p_list, axis=0), [1, 2, 0]), [-1, 9])
                mlp_out = build_mlp(rating_mlp_in, hid_dims, act, drop_rate,
                                    training, scope_name='mlp', bn_first=True)
        return mlp_out

    def train_d(self, batch_uemb, batch_iemb, batch_neg_iemb, batch_content):
        _, d_loss = self.sess.run([self.d_optimizer, self.d_loss],
                                  feed_dict={self.opp_emb: batch_uemb,
                                             self.real_emb: batch_iemb,
                                             self.neg_emb: batch_neg_iemb,
                                             self.content: batch_content,
                                             self.d_training: True,
                                             self.g_training: False,
                                             })
        return [d_loss]

    def train_g(self, batch_uemb, batch_iemb, batch_content):
        _, g_loss, sim_loss = self.sess.run([self.g_optimizer, self.g_loss, self.sim_loss],
                                            feed_dict={self.opp_emb: batch_uemb,
                                                       self.real_emb: batch_iemb,
                                                       self.content: batch_content,
                                                       self.d_training: False,
                                                       self.g_training: True,
                                                       })
        return [g_loss, sim_loss]

    def get_item_emb(self, content, item_emb, warm_item, cold_item):
        out_emb = np.copy(item_emb)
        out_emb[cold_item] = self.sess.run(self.gen_emb, feed_dict={self.content: content[cold_item],
                                                                    self.g_training: False})
        return out_emb

    def get_user_emb(self, user_emb):
        return user_emb

    def get_user_rating(self, uids, iids, uemb, iemb):
        user_rat = self.sess.run(self.user_rating,
                                 feed_dict={self.uemb: uemb[uids],
                                            self.iemb: iemb[iids],
                                            self.d_training: False,
                                            })
        return user_rat

    def get_ranked_rating(self, ratings, k):
        ranked_score, ranked_rat = self.sess.run([self.top_score, self.top_item_index],
                                                 feed_dict={self.rat: ratings,
                                                            self.k: k})
        return ranked_score, ranked_rat
