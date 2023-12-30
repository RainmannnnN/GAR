import os
from metric import ndcg
import utils
import time
import pickle
import argparse
import tensorflow as tf
import numpy as np
import pandas as pd
from pprint import pprint
import random
import torch
import baselines

# other setting
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42, help='Random Seed.')
parser.add_argument('--gpu_id', type=int, default=0)
parser.add_argument('--n_jobs', type=int, default=8, help='Multiprocessing number.')

# dataset
parser.add_argument('--datadir', type=str, default="./data/", help='Director of the dataset.')
parser.add_argument('--dataset', type=str, default="CiteULike", help='Dataset to use.')

# validation & testing
parser.add_argument('--val_interval', type=float, default=1.0)
parser.add_argument('--val_start', type=int, default=0, help='Validation per training batch.')
parser.add_argument('--val_type', type=str, default='cold', choices=['warm', 'cold', 'overall'])
parser.add_argument('--test_batch_us', type=int, default=200)
parser.add_argument('--Ks', nargs='?', default='[20]', help='Output sizes of every layer')
parser.add_argument('--n_test_user', type=int, default=2000)

# base model
parser.add_argument('--embed_meth', type=str, default='node2vec', help='Emebdding method')
parser.add_argument('--embed_loss', type=str, default='BPR', help='Training loss of embedding model.')

# cold-start model training
parser.add_argument('--batch_size', type=int, default=1024, help='Normal batch size.')
parser.add_argument('--train_set', type=str, default='map', choices=['map', 'emb'])
parser.add_argument('--loss', type=str, default='BCE')
parser.add_argument('--max_epoch', type=int, default=10000)
parser.add_argument('--restore', type=str, default="")
parser.add_argument('--patience', type=int, default=10, help='Early stop patience.')

# cold-start model parameter
parser.add_argument('--model', type=str, default='mlp')
parser.add_argument('--alpha', type=float, default=1.0, help='share param')

parser.add_argument('--knn', type=int, default=1, help='Top k in KNN algorithm.')
parser.add_argument('--lamb', type=float, default=0.5, help='param in CLCRec')
parser.add_argument('--temperature', type=float, default=1.0, help='param in CLCRec')

parser.add_argument('--sim_coe', type=float, default=0.1, help='param in GAR')
parser.add_argument('--g_drop', type=float, default=0.1)
parser.add_argument('--d_drop', type=float, default=0.5)
parser.add_argument('--g_act', type=str, default='tanh')
parser.add_argument('--d_act', type=str, default='tanh')

parser.add_argument('--neg', type=float, default=5, help='the times of negative samples to positive samples')
parser.add_argument('--reg', type=float, default=1e-3, )
parser.add_argument('--n_experts', type=int, default=5, help='param in Heater')

args, _ = parser.parse_known_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
args.Ks = eval(args.Ks)
args.model = args.model.upper()
utils.set_seed_tf(args.seed)
pprint(vars(args))
timer = utils.Timer(name='main')
ndcg.init(args)

content_data = np.load(os.path.join(args.datadir, args.dataset, args.dataset + '_item_content.npy'))
# 0 for padding
content_data = np.concatenate([np.zeros([1, content_data.shape[-1]]), content_data], axis=0)
para_dict = pickle.load(open(args.datadir + args.dataset + '/convert_dict.pkl', 'rb'))
train_data = pd.read_csv(args.datadir + args.dataset + '/warm_{}.csv'.format(args.train_set), dtype=np.int64).values

# load embedding
if args.model in ['DEEPMUSIC', 'METAEMB']:
    emb_path = os.path.join(args.datadir, args.dataset, args.embed_meth + '_MLP-{}.npy'.format(args.embed_loss))
else:
    emb_path = os.path.join(args.datadir, args.dataset, "{}.npy".format(args.embed_meth))
user_node_num = max(para_dict['user_array']) + 1
item_node_num = max(para_dict['item_array']) + 1
emb = np.load(emb_path)
user_emb = emb[:user_node_num]
item_emb = emb[user_node_num:]
timer.logging('Embeddings are loaded from {}'.format(emb_path))


# load test set
def get_exclude_pair(u_pair, ts_nei):
    pos_item = np.array(sorted(list(set(para_dict['pos_user_nb'][u_pair[0]]) - set(ts_nei[u_pair[0]]))),
                        dtype=np.int64)
    pos_user = np.array([u_pair[1]] * len(pos_item), dtype=np.int64)
    return np.stack([pos_user, pos_item], axis=1)


def get_exclude_pair_count(ts_user, ts_nei, batch):
    exclude_pair_list = []
    exclude_count = [0]  # 每个 user 有多少 exclude pair
    for i, beg in enumerate(range(0, len(ts_user), batch)):
        end = min(beg + batch, len(ts_user))
        batch_user = ts_user[beg:end]
        batch_range = list(range(end - beg))
        batch_u_pair = tuple(zip(batch_user.tolist(), batch_range))  # (org_id, map_id)

        specialize_get_exclude_pair = lambda x: get_exclude_pair(x, ts_nei)
        exclude_pair = list(map(specialize_get_exclude_pair, batch_u_pair))
        exclude_pair = np.concatenate(exclude_pair, axis=0)

        exclude_pair_list.append(exclude_pair)
        exclude_count.append(exclude_count[i] + len(exclude_pair))

    exclude_pair_list = np.concatenate(exclude_pair_list, axis=0)
    return [exclude_pair_list, exclude_count]


exclude_val_warm = get_exclude_pair_count(para_dict['warm_val_user'][:args.n_test_user], para_dict['warm_val_user_nb'],
                                          args.test_batch_us)
exclude_val_cold = get_exclude_pair_count(para_dict['cold_val_user'][:args.n_test_user], para_dict['cold_val_user_nb'],
                                          args.test_batch_us)
exclude_val_hybrid = get_exclude_pair_count(para_dict['hybrid_val_user'][:args.n_test_user],
                                            para_dict['hybrid_val_user_nb'],
                                            args.test_batch_us)
exclude_test_warm = get_exclude_pair_count(para_dict['warm_test_user'][:args.n_test_user],
                                           para_dict['warm_test_user_nb'],
                                           args.test_batch_us)
exclude_test_cold = get_exclude_pair_count(para_dict['cold_test_user'][:args.n_test_user],
                                           para_dict['cold_test_user_nb'],
                                           args.test_batch_us)
exclude_test_hybrid = get_exclude_pair_count(para_dict['hybrid_test_user'][:args.n_test_user],
                                             para_dict['hybrid_test_user_nb'],
                                             args.test_batch_us)
timer.logging("Loaded excluded pairs for validation and test.")

patience_count = 0
va_metric_max = 0
train_time = 0
val_time = 0
stop_flag = 0
batch_count = 0
item_index = np.arange(item_node_num)

# 模型设置
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # 设置tf模式为按需赠长模式
sess = tf.Session(config=config)
model = eval("baselines.{}".format(args.model))(
    sess, args, emb.shape[-1], content_data.shape[-1])

save_dir = './baselines/model_save/'
os.makedirs(save_dir, exist_ok=True)
save_path = save_dir + args.dataset + '-' + args.model + '-'
param_file = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
save_file = save_path + param_file
args.param_file = param_file
timer.logging('Model will be stored in ' + save_file)
saver = tf.train.Saver()
if len(args.restore) > 1:
    saver.restore(sess, save_path + args.restore)
    timer.logging("Restored model from " + save_path + args.restore)
saver.save(sess, save_file)  # save as a new model

timer.logging("Training model...")
epoch = 0
for epoch in range(1, args.max_epoch + 1):
    if args.loss == 'BPR':
        train_input = utils.bpr_neg_samp(para_dict['warm_user'], len(train_data),
                                         para_dict['emb_user_nb'], para_dict['warm_item'])
    elif args.loss == 'BCE':
        train_input = np.stack(
            utils.negative_sampling(train_data[:, 0],
                                    train_data[:, 1],
                                    args.neg,
                                    para_dict['warm_item']), axis=1)
    elif args.loss == 'SIM':
        train_input = np.random.choice(para_dict['warm_item'], size=(len(train_data),), replace=True)
    else:
        raise ValueError("Loss not found!")

    # 保证每一个 batch 一样大小，舍去最后一个不完整的 batch
    n_batch = len(train_input) // args.batch_size
    for beg in range(0, len(train_input) - args.batch_size, args.batch_size):
        end = beg + args.batch_size
        batch_count += 1
        t_train_begin = time.time()
        batch_lbs = train_input[beg: end]

        if 'GAR' in args.model:
            d_loss = model.train_d(user_emb[batch_lbs[:, 0]],
                                   item_emb[batch_lbs[:, 1]],
                                   item_emb[batch_lbs[:, 2]],
                                   content_data[batch_lbs[:, 1]])
            g_loss = model.train_g(user_emb[batch_lbs[:, 0]],
                                   item_emb[batch_lbs[:, 1]],
                                   content_data[batch_lbs[:, 1]])
            loss = sum(d_loss + g_loss)
        elif args.loss == 'SIM':
            loss = model.train_sim(content_data[batch_lbs], item_emb[batch_lbs])
        elif args.loss == 'BPR':
            loss = model.train_bpr(content_data[batch_lbs[:, 1]],
                                   item_emb[batch_lbs[:, 1]],
                                   content_data[batch_lbs[:, 2]],
                                   item_emb[batch_lbs[:, 2]],
                                   user_emb[batch_lbs[:, 0]],
                                   )
        elif args.loss == 'BCE':
            loss = model.train_bce(content_data[batch_lbs[:, 1]],
                                   item_emb[batch_lbs[:, 1]],
                                   user_emb[batch_lbs[:, 0]],
                                   batch_lbs[:, 2])
        t_train_end = time.time()
        train_time += t_train_end - t_train_begin

        # Validation - interval can be float
        if (batch_count % int(n_batch * args.val_interval) == 0) and (epoch >= args.val_start):
            t_val_begin = time.time()
            num_val = batch_count // args.val_interval

            gen_user_emb = model.get_user_emb(user_emb)
            gen_item_emb = model.get_item_emb(content_data, item_emb,
                                              para_dict['warm_item'], para_dict['cold_item'])

            if args.val_type == 'warm':  # warm val
                va_metric, _ = ndcg.test(model.get_ranked_rating,
                                         lambda u: model.get_user_rating(u, item_index, gen_user_emb, gen_item_emb),
                                         ts_nei=para_dict['warm_val_user_nb'],
                                         ts_user=para_dict['warm_val_user'][:args.n_test_user],
                                         masked_items=para_dict['cold_item'],
                                         exclude_pair_cnt=exclude_val_warm,
                                         )
            elif args.val_type == 'cold':  # cold val
                va_metric, _ = ndcg.test(model.get_ranked_rating,
                                         lambda u: model.get_user_rating(u, item_index, gen_user_emb, gen_item_emb),
                                         ts_nei=para_dict['cold_val_user_nb'],
                                         ts_user=para_dict['cold_val_user'][:args.n_test_user],
                                         masked_items=para_dict['warm_item'],
                                         exclude_pair_cnt=exclude_val_cold,
                                         )
            elif args.val_type == 'overall':  # overall val
                va_metric, _ = ndcg.test(model.get_ranked_rating,
                                         lambda u: model.get_user_rating(u, item_index, gen_user_emb, gen_item_emb),
                                         ts_nei=para_dict['hybrid_val_user_nb'],
                                         ts_user=para_dict['hybrid_val_user'][:args.n_test_user],
                                         masked_items=None,
                                         exclude_pair_cnt=exclude_val_hybrid,
                                         )

            va_metric_current = va_metric['ndcg'][0]
            if va_metric_current > va_metric_max:
                va_metric_max = va_metric_current
                saver.save(sess, save_file)
                patience_count = 0
            else:
                patience_count += 1
                if patience_count > args.patience:
                    stop_flag = 1
                    break

            t_val_end = time.time()
            val_time += t_val_end - t_val_begin
            timer.logging('Epo%d(%d/%d) Loss:%.4f|va_metric:%.4f|Best:%.4f|Time_Tr:%.2fs|Val:%.2fs' %
                          (epoch, patience_count, args.patience, loss,
                           va_metric_current, va_metric_max, train_time, val_time))
    if stop_flag:
        break
timer.logging("Finish training model at epoch {}.".format(epoch))

""" Test """
saver.restore(sess, save_file)
gen_user_emb = model.get_user_emb(user_emb)
gen_item_emb = model.get_item_emb(content_data, item_emb, para_dict['warm_item'], para_dict['cold_item'])
emb_store_path = os.path.join(args.datadir,
                              args.dataset,
                              "{}_{}{}.npy".format(args.embed_meth, args.model,
                                                   f'-{args.loss}' if args.model in 'MLP' else ""))
np.save(emb_store_path, np.concatenate([gen_user_emb, gen_item_emb], axis=0))
timer.logging("Tranformed embeddings are stored in {}.".format(emb_store_path))

# cold recommendation
cold_res, _ = ndcg.test(model.get_ranked_rating,
                        lambda u: model.get_user_rating(u, item_index, gen_user_emb, gen_item_emb),
                        ts_nei=para_dict['cold_test_user_nb'],
                        ts_user=para_dict['cold_test_user'][:args.n_test_user],
                        masked_items=para_dict['warm_item'],
                        exclude_pair_cnt=exclude_test_cold,
                        )
timer.logging(
    'Cold-start recommendation result@{}: PRE, REC, NDCG: {:.4f}, {:.4f}, {:.4f}'.format(
        args.Ks[0], cold_res['precision'][0], cold_res['recall'][0], cold_res['ndcg'][0]))

# warm recommendation
warm_res, warm_dist = ndcg.test(model.get_ranked_rating,
                                lambda u: model.get_user_rating(u, item_index, gen_user_emb, gen_item_emb),
                                ts_nei=para_dict['warm_test_user_nb'],
                                ts_user=para_dict['warm_test_user'][:args.n_test_user],
                                masked_items=para_dict['cold_item'],
                                exclude_pair_cnt=exclude_test_warm,
                                )
timer.logging("Warm recommendation result@{}: PRE, REC, NDCG: {:.4f}, {:.4f}, {:.4f}".format(
    args.Ks[0], warm_res['precision'][0], warm_res['recall'][0], warm_res['ndcg'][0]))

# hybrid recommendation
hybrid_res, _ = ndcg.test(model.get_ranked_rating,
                          lambda u: model.get_user_rating(u, item_index, gen_user_emb, gen_item_emb),
                          ts_nei=para_dict['hybrid_test_user_nb'],
                          ts_user=para_dict['hybrid_test_user'][:args.n_test_user],
                          masked_items=None,
                          exclude_pair_cnt=exclude_test_hybrid,
                          )
timer.logging("Hybrid recommendation result@{}: PRE, REC, NDCG: {:.4f}, {:.4f}, {:.4f}".format(
    args.Ks[0], hybrid_res['precision'][0], hybrid_res['recall'][0], hybrid_res['ndcg'][0]))

# 保存测试结果
sess.close()  # 关闭 session
result_dir = './baselines/result/'
os.makedirs(result_dir, exist_ok=True)
with open(result_dir + f'{args.model}.txt', 'a') as f:
    f.write(str(vars(args)))
    f.write(' | ')
    for i in range(len(args.Ks)):
        f.write('%.4f %.4f %.4f ' % (cold_res['precision'][i], cold_res['recall'][i], cold_res['ndcg'][i]))
    f.write(' | ')
    for i in range(len(args.Ks)):
        f.write('%.4f %.4f %.4f ' % (warm_res['precision'][i], warm_res['recall'][i], warm_res['ndcg'][i]))
    f.write(' | ')
    for i in range(len(args.Ks)):
        f.write('%.4f %.4f %.4f ' % (hybrid_res['precision'][i], hybrid_res['recall'][i], hybrid_res['ndcg'][i]))
    f.write('\n')
