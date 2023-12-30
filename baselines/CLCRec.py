import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import activations


class CLCREC:
    def __init__(self, sess, args, emb_dim, content_dim):
        self.sess = sess
        self.emb_dim = emb_dim  # input embedding dimension
        self.content_dim = content_dim
        self.reg = args.reg  # coefficient of regularization
        self.lr = 1e-3
        regularizer = keras.regularizers.l2(self.reg)
        initializer = keras.initializers.truncated_normal(stddev=0.01)
        self.temperature = args.temperature
        self.lamb = args.lamb
        self.labels = tf.one_hot(indices=list(range(args.batch_size)), depth=args.batch_size)

        self.is_training = tf.placeholder(tf.bool, name='is_training')
        self.target = tf.placeholder(tf.float32, shape=[None], name='target')
        self.Uin = tf.placeholder(tf.float32, shape=[None, self.emb_dim], name='u_emb')
        self.Vin = tf.placeholder(tf.float32, shape=[None, self.emb_dim], name='v_emb')

        # get fu and fv
        self.Vcontent = tf.placeholder(tf.float32, shape=[None, self.content_dim])
        self.dropout_item_indicator = tf.placeholder(tf.float32, [None, 1], name='dropout_flag')

        feat_encoder_1 = layers.Dense(256,
                                      kernel_regularizer=regularizer,
                                      kernel_initializer=initializer)
        feat_encoder_2 = layers.Dense(self.emb_dim,
                                      kernel_regularizer=regularizer,
                                      kernel_initializer=initializer)
        feat_representation = feat_encoder_2(activations.tanh(feat_encoder_1(self.Vcontent)))
        reg_loss_feat = tf.reduce_sum(feat_encoder_1.losses) + tf.reduce_sum(feat_encoder_2.losses)

        # get hat(U) and hat(V)
        user_encoder_1 = layers.Dense(self.emb_dim,
                                      kernel_regularizer=regularizer,
                                      kernel_initializer=initializer)
        user_encoder_2 = layers.Dense(self.emb_dim,
                                      kernel_regularizer=regularizer,
                                      kernel_initializer=initializer)
        self.U_embedding = user_encoder_2(activations.tanh(user_encoder_1(self.Uin)))
        reg_loss_user = tf.reduce_sum(user_encoder_1.losses) + tf.reduce_sum(user_encoder_2.losses)

        item_encoder_1 = layers.Dense(self.emb_dim,
                                      kernel_regularizer=regularizer,
                                      kernel_initializer=initializer)
        item_encoder_2 = layers.Dense(self.emb_dim,
                                      kernel_regularizer=regularizer,
                                      kernel_initializer=initializer)
        cf_representation = item_encoder_2(activations.tanh(item_encoder_1(self.Vin)))
        reg_loss_item = tf.reduce_sum(item_encoder_1.losses) + tf.reduce_sum(item_encoder_2.losses)
        self.V_embedding = cf_representation * self.dropout_item_indicator + feat_representation * (1 - self.dropout_item_indicator)

        with tf.variable_scope("loss"):
            # normalization can prevent nan loss
            self.contrastive_loss_1 = self.loss_contrastive(self.U_embedding, self.V_embedding, self.temperature) * (1 - self.lamb)
            self.contrastive_loss_2 = self.loss_contrastive(cf_representation, feat_representation, self.temperature) * self.lamb
            self.reg_loss = reg_loss_user + reg_loss_item + reg_loss_feat
            self.loss = self.contrastive_loss_1 + self.contrastive_loss_2 + self.reg * self.reg_loss

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

    def loss_contrastive(self, tensor_anchor, tensor_all, temperature):
        all_logits = tf.matmul(tensor_anchor, tensor_all, transpose_b=True) / temperature
        contrastive_loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(logits=all_logits, labels=self.labels),
            axis=0,
        )
        return contrastive_loss

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
        _, loss, con_loss_1, con_loss_2, reg_loss = self.sess.run([self.optimizer, self.loss, self.contrastive_loss_1,
                                                                   self.contrastive_loss_2, self.reg_loss],
                                                                  feed_dict=_train_dict)
        # if np.isnan(loss):
        #     print("contrastive loss 1:", con_loss_1)
        #     print("contrastive loss 2:", con_loss_2)
        #     print("reg loss:", reg_loss)
        #     exit()
        # return [loss, con_loss_1, con_loss_2, reg_loss]
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