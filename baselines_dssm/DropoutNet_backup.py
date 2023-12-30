import tensorflow as tf
import numpy as np
from copy import deepcopy
from tensorflow.keras.layers import Activation


class DROPOUTNET_BACKUP(object):
    def __init__(self, args, session, feature_embedding_dim,
                 user_feature_columns, item_feature_columns, item_to_frequency,
                 user_dnn_hidden_units=(128, 32), item_dnn_hidden_units=(128, 32),
                 dnn_activation='relu', dnn_use_bn=False,
                 lr=1e-3, l2_reg_dnn=1e-4, l2_reg_embedding=0.,
                 dnn_dropout=0., temperature=0.05,
                 ):
        self.sess = session
        self.feature_embdedding_dim = feature_embedding_dim
        self.emb_drop_rate = args.emb_drop_rate
        assert user_dnn_hidden_units[-1] == item_dnn_hidden_units[-1]

        self.training = tf.placeholder(tf.bool, name='training_or_not')
        self.target = tf.placeholder(tf.float32, shape=[None], name='target')
        self.dropout_item_indicator = tf.placeholder(tf.float32, shape=[None, 1], name='drop_i_indicator')
        with tf.variable_scope('Emb', reuse=False):
            self.feature_columns = deepcopy(user_feature_columns)
            self.feature_columns.update(item_feature_columns)
            self.emb_dict = self.build_embedding_matrix(self.feature_columns, l2_reg_embedding)
        # build user input
        self.user_feature_columns = user_feature_columns
        self.user_features = self.build_input_features(self.user_feature_columns)
        self.user_input = self.feature_to_emb(self.user_features, user_feature_columns, self.emb_dict)
        # build user tower
        with tf.variable_scope('user_tower', reuse=False):
            self.user_emb = self.build_dnn(self.user_input, user_dnn_hidden_units, dnn_activation,
                                           l2_reg_dnn, dnn_dropout, dnn_use_bn)
            self.user_emb = tf.nn.l2_normalize(self.user_emb, axis=-1)

        # build item input
        self.item_feature_columns = item_feature_columns
        self.item_features = self.build_input_features(self.item_feature_columns)
        self.item_id_emb, self.item_content_emb = self.feature_to_emb(
            self.item_features, item_feature_columns, self.emb_dict)

        self.train_item_input = tf.concat(
            [self.item_id_emb * self.dropout_item_indicator, self.item_content_emb], axis=1)
        self.test_item_input = tf.concat([self.item_id_emb, self.item_content_emb], axis=1)
        # build item tower
        with tf.variable_scope('item_tower', reuse=False):
            self.train_item_emb = self.build_dnn(self.train_item_input, item_dnn_hidden_units, dnn_activation,
                                                 l2_reg_dnn, dnn_dropout, dnn_use_bn)
            self.train_item_emb = tf.nn.l2_normalize(self.train_item_emb, axis=-1)
        with tf.variable_scope('item_tower', reuse=True):
            self.test_item_emb = self.build_dnn(self.test_item_input, item_dnn_hidden_units, dnn_activation,
                                                l2_reg_dnn, dnn_dropout, dnn_use_bn)
            self.test_item_emb = tf.nn.l2_normalize(self.test_item_emb, axis=-1)

        with tf.variable_scope("loss"):
            self.preds = tf.reduce_sum(tf.multiply(self.user_emb, self.train_item_emb), axis=-1)
            self.base_loss = tf.reduce_mean(tf.squared_difference(self.preds, self.target))
            # self.base_loss = tf.reduce_mean(
            #     tf.nn.sigmoid_cross_entropy_with_logits(logits=self.preds, labels=self.target))
            self.l2_loss = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
            self.loss = self.base_loss + self.l2_loss
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.optimizer = tf.train.AdamOptimizer(lr).minimize(self.loss)

        # get user rating through dot
        self.rat_uemb = tf.placeholder(tf.float32, [None, user_dnn_hidden_units[-1]], name='user_embedding')
        self.rat_iemb = tf.placeholder(tf.float32, [None, item_dnn_hidden_units[-1]], name='item_embedding')
        self.user_rating = tf.matmul(self.rat_uemb, self.rat_iemb, transpose_b=True)

        # rank user rating
        self.rating_to_rank = tf.placeholder(tf.float32, [None, None], name='rating_to_rank')
        self.k = tf.placeholder(tf.int32, name='topK')
        self.top_score, self.top_item_index = tf.nn.top_k(self.rating_to_rank, k=self.k)

        self.sess.run(tf.global_variables_initializer())
        print([v.name for v in tf.trainable_variables()])

    def build_embedding_matrix(self, feature_columns, l2_reg):
        embedding_dict = {}
        for name, info in feature_columns.items():
            if info['type'] in 'ctg':
                continue
            emb_matrix = tf.get_variable('emb_' + name,
                                         [info['num'] + 1, self.feature_embdedding_dim],
                                         initializer=tf.random_normal_initializer(stddev=1e-4),
                                         regularizer=tf.contrib.layers.l2_regularizer(l2_reg),
                                         )
            embedding_dict[name] = emb_matrix
        return embedding_dict

    def build_input_features(self, feature_columns) -> dict:
        input_features = {}
        for name, info in feature_columns.items():
            if info['type'] in 'spr':
                input_features[name] = tf.placeholder(tf.int64, [None], 'in_' + name)
            elif info['type'] in 'seq':
                input_features[name] = tf.placeholder(tf.int64, [None, info['max_length']], 'in_' + name)
            elif info['type'] in 'ctg':
                input_features[name] = tf.placeholder(tf.float32, [None, info['dim']], 'in_' + name)
            else:
                raise TypeError("Invalid feature column type,got", info['type'])
        return input_features

    def feature_to_emb(self, features, feature_columns, embedding_matrix_dict):
        # todo 分出 id emb 和 content
        spr_feature_columns = dict(filter(lambda x: x[1]['type'] in 'spr', feature_columns.items()))
        seq_feature_columns = dict(filter(lambda x: x[1]['type'] in 'seq', feature_columns.items()))
        ctg_feature_columns = dict(filter(lambda x: x[1]['type'] in 'ctg', feature_columns.items()))
        spr_embedding_dict = {}
        id_embedding = None
        for name, info in spr_feature_columns.items():
            lookup_idx = features[name]
            spr_embedding = tf.nn.embedding_lookup(embedding_matrix_dict[name], lookup_idx)
            # 分出 item id embedding，user 不需要，只做 item embedding
            if name in 'item':
                id_embedding = spr_embedding
                continue
            spr_embedding_dict[name] = spr_embedding
        seq_embedding_dict = {}
        for name, info in seq_feature_columns.items():
            lookup_idx = features[name]
            seq_embedding = tf.nn.embedding_lookup(embedding_matrix_dict[name], lookup_idx)
            seq_embedding_dict[name] = tf.reduce_mean(seq_embedding, -2)
        ctg_feature_dict = {}
        for name, info in ctg_feature_columns.items():
            ctg_feature_dict[name] = features[name]
        content_embedding = list(spr_embedding_dict.values()) + \
                            list(seq_embedding_dict.values()) + \
                            list(ctg_feature_dict.values())
        content_embedding = tf.concat(content_embedding, axis=-1)
        if id_embedding is None:
            return content_embedding
        else:
            return id_embedding, content_embedding

    def build_dnn(self, dnn_in, hidden_units, activation, l2_reg, drop_rate, use_bn=False):
        hidden = dnn_in
        for i in range(1, len(hidden_units) + 1):
            hidden = tf.layers.dense(hidden, hidden_units[i - 1], name=f'linear_{i}',
                                     kernel_initializer=tf.glorot_uniform_initializer(),
                                     kernel_regularizer=tf.contrib.layers.l2_regularizer(l2_reg))
            if i == len(hidden_units):  # last linear
                break
            hidden = Activation(activation)(hidden)
            if use_bn:
                hidden = tf.layers.batch_normalization(hidden, training=self.training, name=f'bn_{i}')
            if drop_rate > 1e-5:
                hidden = tf.layers.dropout(hidden, rate=drop_rate, training=self.training, name=f'drop_{i}')
        return hidden

    def train_bce(self, features_dict):
        input_dict = {self.user_features[col]: features_dict[col].to_list() for col in self.user_feature_columns}
        input_dict.update({self.item_features[col]: features_dict[col].to_list() for col in self.item_feature_columns})
        drop_indicator = np.random.uniform(size=[len(features_dict['label']), 1])
        drop_indicator = (drop_indicator > self.emb_drop_rate).astype(np.float32)
        input_dict.update({self.dropout_item_indicator: drop_indicator,
                           self.training: True,
                           self.target: features_dict['label']})
        _, loss, base_loss, l2_loss = self.sess.run([self.optimizer, self.loss, self.base_loss, self.l2_loss],
                                                    feed_dict=input_dict)
        return loss

    def get_user_emb(self, user_feat_dict):
        input_dict = {self.user_features[col]: user_feat_dict[col].to_list() for col in self.user_feature_columns}
        input_dict.update({self.training: False})
        user_emb = self.sess.run(self.user_emb, feed_dict=input_dict)
        # map user_id to index
        ret_user_emb = np.zeros((self.feature_columns['user']['num'] + 1, user_emb.shape[1]), dtype=np.float32)
        ret_user_emb[user_feat_dict['user']] = user_emb
        return ret_user_emb

    def get_item_emb(self, item_feat_dict, warm_item, cold_item):
        input_dict = {self.item_features[col]: item_feat_dict[col].to_list() for col in self.item_feature_columns}
        input_dict.update({self.training: False})
        drop_indicator = np.ones((self.feature_columns['item']['num'] + 1, 1))
        drop_indicator[cold_item] = 0
        drop_indicator = drop_indicator[item_feat_dict['item']]
        input_dict.update({self.dropout_item_indicator: drop_indicator})
        item_emb = self.sess.run(self.test_item_emb, feed_dict=input_dict)
        # map item_id to index
        ret_item_emb = np.zeros((self.feature_columns['item']['num'] + 1, item_emb.shape[1]), dtype=np.float32)
        ret_item_emb[item_feat_dict['item']] = item_emb
        return ret_item_emb

    def get_user_rating(self, uids, uemb_mat, iemb_mat):
        user_rating = self.sess.run(self.user_rating,
                                    feed_dict={self.rat_uemb: uemb_mat[uids],
                                               self.rat_iemb: iemb_mat})
        return user_rating

    def get_ranked_rating(self, rating, k):
        ranked_score, ranked_index = self.sess.run([self.top_score, self.top_item_index],
                                                   feed_dict={self.rating_to_rank: rating,
                                                              self.k: k})
        return ranked_score, ranked_index
