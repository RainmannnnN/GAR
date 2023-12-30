import numpy as np
import tensorflow as tf
from copy import deepcopy


class MAP_KNN(object):
    def __init__(self, sess, args, emb_dim, content_dim):
        self.sess = sess
        self.emb_dim = emb_dim
        self.content_dim = content_dim
        self.batch_size = 1024
        self.placeholder = tf.get_variable('W', shape=[100, 100],
                                           initializer=tf.truncated_normal_initializer(stddev=0.01))
        self.knn = args.knn

        # content similarity
        self.warm_content = tf.placeholder(tf.float32, [None, self.content_dim], name='warm_content')
        self.cold_content = tf.placeholder(tf.float32, [None, self.content_dim], name='cold_content')
        self.content_similarity = tf.matmul(tf.math.l2_normalize(self.cold_content),  # cosine similarity
                                            tf.transpose(tf.math.l2_normalize(self.warm_content)))
        self.top_similar_values, self.top_similar_warm_item_index = tf.nn.top_k(self.content_similarity,
                                                                                k=self.knn)

        # softmax pooling
        self.weight = tf.placeholder(tf.float32, [None, None])
        self.input_to_pooling = tf.placeholder(tf.float32, [None, None, self.emb_dim])

        self.softmax_weight = tf.nn.softmax(self.weight)
        self.pooling_out = tf.reduce_sum(tf.expand_dims(self.softmax_weight, axis=2) * self.input_to_pooling, axis=1)

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
        g_loss = 0.0
        return g_loss

    def train_mapping(self, batch_uemb, batch_iemb):
        return None

    def get_user_rating(self, uids, iids, uemb, iemb):
        user_rat = self.sess.run(self.user_rating,
                                 feed_dict={self.uemb: uemb[uids],
                                            self.iemb: iemb[iids], })
        return user_rat

    # get generated embeddings
    def get_item_emb(self, content, item_emb, warm_index, cold_index):
        batches = [(begin, min(begin + self.batch_size, len(cold_index)))
                   for begin in range(0, len(cold_index), self.batch_size)]
        top_warm_index_in_warm = []
        top_similar_values = []

        for begin, end in batches:
            batch_cold_index = cold_index[begin: end]
            batch_top_similar_values, batch_top_warm_index_in_warm = self.sess.run(
                [self.top_similar_values, self.top_similar_warm_item_index],
                feed_dict={self.cold_content: content[batch_cold_index],
                           self.warm_content: content[warm_index],
                           })
            top_warm_index_in_warm.append(batch_top_warm_index_in_warm)
            top_similar_values.append(batch_top_similar_values)
        top_warm_index_in_warm = np.concatenate(top_warm_index_in_warm, axis=0)  # (colds, k)
        top_similar_values = np.concatenate(top_similar_values, axis=0)

        top_warm_index_in_warm = np.reshape(top_warm_index_in_warm, (-1,))  # (colds * k)
        top_warm_index = warm_index[top_warm_index_in_warm]  # (colds * k, emb)
        k_cold_emb = np.reshape(item_emb[top_warm_index], (len(cold_index), self.knn, self.emb_dim))  # (colds, k, emb)

        cold_emb = self.sess.run(self.pooling_out, feed_dict={self.weight: top_similar_values,
                                                              self.input_to_pooling: k_cold_emb})
        copy_item_emb = deepcopy(item_emb)  # 避免修改输入
        copy_item_emb[cold_index] = cold_emb
        return copy_item_emb

    def get_user_emb(self, user_emb):
        return user_emb

    def get_ranked_rating(self, ratings, k):
        ranked_score, ranked_rat = self.sess.run([self.top_score, self.top_item_index],
                                   feed_dict={self.rat: ratings,
                                              self.k: k})
        return ranked_score, ranked_rat