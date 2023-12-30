import tensorflow as tf
import numpy as np


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

class DROPOUTNET:
    def __init__(self, sess, args, emb_dim, content_dim):
        self.sess = sess
        self.emb_dim = emb_dim  # input embedding dimension
        self.transform_layers = [3 * self.emb_dim, 3 * self.emb_dim]  # output dimension
        self.content_dim = content_dim
        self.reg = 1e-3  # coefficient of regularization
        self.lr = 1e-3
        self.d_drop = 0.0
        self.d_layer = [256, 256, 1]
        self.d_act = 'relu'

        self.is_training = tf.placeholder(tf.bool, name='is_training')
        self.target = tf.placeholder(tf.float32, shape=[None, 1], name='target')
        self.Uin = tf.placeholder(tf.float32, shape=[None, 3, self.emb_dim], name='u_emb')
        self.Vin = tf.placeholder(tf.float32, shape=[None, 3, self.emb_dim], name='v_emb')
        self.Vcontent = tf.placeholder(tf.float32, shape=[None, self.content_dim])
        self.dropout_item_indicator = tf.placeholder(tf.float32, [None, 1], name='dropout_flag')

        # get fu and fv
        self.gen_emb = build_mlp(self.Vcontent, self.transform_layers, 'tanh', 0.0, self.is_training, 'G')
        self.gen_emb = tf.reshape(self.gen_emb, [-1, 3, self.emb_dim])

        # get hat(U) and hat(V)
        self.U_embedding = self.Uin
        self.V_embedding = tf.tile(tf.expand_dims(self.dropout_item_indicator, axis=1), [1, 3, 1]) * self.Vin
        # dropped item 跟 not dropped item 会有不同的输入分布
        self.V_embedding += self.gen_emb

        self.preds = self.build_discriminator(self.U_embedding, self.V_embedding,
                                            self.d_layer, self.d_act, self.d_drop,
                                            training=self.is_training, reuse=False, rating=False)
        # self.target_loss = tf.reduce_mean(tf.squared_difference(self.preds, self.target))
        self.target_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.preds, labels=self.target))
        self.reg_loss = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, scope='G'))
        self.loss = self.target_loss + self.reg_loss
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

        # get user rating through dot
        self.uemb = tf.placeholder(tf.float32, [None, 3, self.emb_dim], name='user_embedding')
        self.iemb = tf.placeholder(tf.float32, [None, 3, self.emb_dim], name='item_embedding')
        self.user_rating = self.build_discriminator(self.uemb, self.iemb,
                                                    self.d_layer, self.d_act, self.d_drop,
                                                    training=self.is_training, reuse=True, rating=True)
        self.user_rating = tf.reshape(self.user_rating, [tf.shape(self.uemb)[0], -1])

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
            self.target: target.reshape([-1, 1]),
            self.dropout_item_indicator: drop_indicator
        }
        _, loss = self.sess.run([self.optimizer, self.loss], feed_dict=_train_dict)
        return loss

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
                                    training, scope_name='mlp', bn=False, bn_first=False)
            else:
                rating_p_list = [tf.matmul(
                    tf.transpose(uemb_bn, [1, 0, 2]),
                    tf.transpose(tf.roll(iemb_bn, shift=_, axis=1), [1, 2, 0])) for _ in range(3)]
                rating_mlp_in = tf.reshape(tf.transpose(tf.concat(rating_p_list, axis=0), [1, 2, 0]), [-1, 9])
                mlp_out = build_mlp(rating_mlp_in, hid_dims, act, drop_rate,
                                    training, scope_name='mlp', bn=False, bn_first=False)
        return mlp_out

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
