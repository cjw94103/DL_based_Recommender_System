import argparse
import torch
import scipy.sparse as sp
import numpy as np
import metrics
import multiprocessing
import heapq
import matplotlib.pyplot as plt

from models import NGCF
from tqdm import tqdm

## argparse
def list_of_ints(arg):
    return list(map(int, arg.split(',')))

def list_of_floats(arg):
    return list(map(float, arg.split(',')))

parser = argparse.ArgumentParser()

# path preprocessed dataset
parser.add_argument("--testset_path", type=str, help="save path testset npy", default="./data/amazon_book/testset.npy")

parser.add_argument("--user_item_info_path", type=str, help="save path user_item_info npy", default="./data/amazon_book/user_item_info.npy")

parser.add_argument("--adj_mat_path", type=str, help="save path adj_mat npz", default="./data/amazon_book/s_adj_mat.npz")
parser.add_argument("--norm_adj_mat_path", type=str, help="save path norm_adj_mat npz", default="./data/amazon_book/s_norm_adj_mat.npz")
parser.add_argument("--mean_mat_path", type=str, help="save path mean_adj_mat npz", default="./data/amazon_book/s_mean_adj_mat.npz")

# gpu parameter
parser.add_argument("--gpu_id", type=str, help="your gpu id", default="0")

# model parameter
parser.add_argument("--batch_size", type=int, help="num of batch size", default=512)
parser.add_argument("--embed_size", type=int, help="number of embedding size for user and item", default=64)
parser.add_argument("--node_dropout", type=float, help="node dropout ratio", default=0.1)
parser.add_argument("--mess_dropout", type=list_of_floats, help="message dropout ratio each layer", default="0.1, 0.1, 0.1")
parser.add_argument("--layer_size", type=list_of_ints, help="output size of embed prop. layer", default="64, 64, 64")
parser.add_argument("--weight_decay", type=float, help="l2 decay", default=1e-5)

# trained model path
parser.add_argument("--model_save_path", type=str, help="your model save path", default="./model_result/NGCF_AmazonBook.pth")

args = parser.parse_args()

## load testset
# load testset
testset = np.load(args.testset_path, allow_pickle=True).item()

# load adj mat
norm_adj = sp.load_npz(args.norm_adj_mat_path)

## load user_item_info
user_item_info = np.load(args.user_item_info_path, allow_pickle=True).item()

USR_NUM, ITEM_NUM = user_item_info['n_users'], user_item_info['n_items']
N_TEST = user_item_info['n_test']
BATCH_SIZE = args.batch_size
Ks = [20]

# for parallel validation
cores = multiprocessing.cpu_count() // 2

## get trained model
device = torch.device('cuda:' + str(args.gpu_id))

model = NGCF(n_user=user_item_info['n_users'],
            n_item=user_item_info['n_items'],
            norm_adj=norm_adj,
            device=device,
            embed_size=args.embed_size,
            batch_size=args.batch_size,
            node_dropout=args.node_dropout,
            mess_dropout=args.mess_dropout,
            layer_size=args.layer_size,
            weight_decay=args.weight_decay)

weights = torch.load(args.model_save_path)
model.load_state_dict(weights)
model.to(device)

## evaluate function
def ranklist_by_heapq(user_pos_test, test_items, rating, Ks):
    item_score = {}
    for i in test_items:
        item_score[i] = rating[i]

    K_max = max(Ks)
    K_max_item_score = heapq.nlargest(K_max, item_score, key=item_score.get)

    r = []
    for i in K_max_item_score:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    auc = 0.
    return r, auc

def get_auc(item_score, user_pos_test):
    item_score = sorted(item_score.items(), key=lambda kv: kv[1])
    item_score.reverse()
    item_sort = [x[0] for x in item_score]
    posterior = [x[1] for x in item_score]

    r = []
    for i in item_sort:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    auc = metrics.auc(ground_truth=r, prediction=posterior)
    return auc

def ranklist_by_sorted(user_pos_test, test_items, rating, Ks):
    item_score = {}
    for i in test_items:
        item_score[i] = rating[i]

    K_max = max(Ks)
    K_max_item_score = heapq.nlargest(K_max, item_score, key=item_score.get)

    r = []
    for i in K_max_item_score:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    auc = get_auc(item_score, user_pos_test)
    return r, auc

def get_performance(user_pos_test, r, auc, Ks):
    precision, recall, ndcg, hit_ratio = [], [], [], []

    for K in Ks:
        precision.append(metrics.precision_at_k(r, K))
        recall.append(metrics.recall_at_k(r, K, len(user_pos_test)))
        ndcg.append(metrics.ndcg_at_k(r, K, user_pos_test))
        hit_ratio.append(metrics.hit_at_k(r, K))

    return {'recall': np.array(recall), 'precision': np.array(precision),
            'ndcg': np.array(ndcg), 'hit_ratio': np.array(hit_ratio), 'auc': auc}

def test_one_user(x):
    # user u's ratings for user u
    rating = x[0]
    #uid
    u = x[1]
    #user u's items in the training set
    try:
        training_items = data_generator.train_items[u]
    except Exception:
        training_items = []
    #user u's items in the test set
    user_pos_test = testset[u]

    all_items = set(range(ITEM_NUM))

    test_items = list(all_items - set(training_items))

    r, auc = ranklist_by_heapq(user_pos_test, test_items, rating, Ks)

    # if args.test_flag == 'part':
    #     r, auc = ranklist_by_heapq(user_pos_test, test_items, rating, Ks)
    # else:
    #     r, auc = ranklist_by_sorted(user_pos_test, test_items, rating, Ks)

    return get_performance(user_pos_test, r, auc, Ks)


def test(model, users_to_test, drop_flag=False, batch_test_flag=False):
    result = {'precision': np.zeros(len(Ks)), 'recall': np.zeros(len(Ks)), 'ndcg': np.zeros(len(Ks)),
              'hit_ratio': np.zeros(len(Ks)), 'auc': 0.}

    pool = multiprocessing.Pool(cores)

    u_batch_size = BATCH_SIZE * 2
    i_batch_size = BATCH_SIZE

    test_users = users_to_test
    n_test_users = len(test_users)
    n_user_batchs = n_test_users // u_batch_size + 1

    count = 0
    
    for u_batch_id in tqdm(range(n_user_batchs)):
        start = u_batch_id * u_batch_size
        end = (u_batch_id + 1) * u_batch_size

        user_batch = test_users[start: end]

        if batch_test_flag:
            # batch-item test
            n_item_batchs = ITEM_NUM // i_batch_size + 1
            rate_batch = np.zeros(shape=(len(user_batch), ITEM_NUM))

            i_count = 0
            for i_batch_id in range(n_item_batchs):
                i_start = i_batch_id * i_batch_size
                i_end = min((i_batch_id + 1) * i_batch_size, ITEM_NUM)

                item_batch = range(i_start, i_end)

                if drop_flag == False:
                    u_g_embeddings, pos_i_g_embeddings, _ = model(user_batch,
                                                                  item_batch,
                                                                  [],
                                                                  drop_flag=False)
                    i_rate_batch = model.rating(u_g_embeddings, pos_i_g_embeddings).detach().cpu()
                else:
                    u_g_embeddings, pos_i_g_embeddings, _ = model(user_batch,
                                                                  item_batch,
                                                                  [],
                                                                  drop_flag=True)
                    i_rate_batch = model.rating(u_g_embeddings, pos_i_g_embeddings).detach().cpu()

                rate_batch[:, i_start: i_end] = i_rate_batch
                i_count += i_rate_batch.shape[1]

            assert i_count == ITEM_NUM

        else:
            # all-item test
            item_batch = range(ITEM_NUM)

            if drop_flag == False:
                u_g_embeddings, pos_i_g_embeddings, _ = model(user_batch,
                                                              item_batch,
                                                              [],
                                                              drop_flag=False)
                rate_batch = model.rating(u_g_embeddings, pos_i_g_embeddings).detach().cpu()
            else:
                u_g_embeddings, pos_i_g_embeddings, _ = model(user_batch,
                                                              item_batch,
                                                              [],
                                                              drop_flag=True)
                rate_batch = model.rating(u_g_embeddings, pos_i_g_embeddings).detach().cpu()

        user_batch_rating_uid = zip(rate_batch.numpy(), user_batch)
        batch_result = pool.map(test_one_user, user_batch_rating_uid)
        count += len(batch_result)

        for re in batch_result:
            result['precision'] += re['precision']/n_test_users
            result['recall'] += re['recall']/n_test_users
            result['ndcg'] += re['ndcg']/n_test_users
            result['hit_ratio'] += re['hit_ratio']/n_test_users
            result['auc'] += re['auc']/n_test_users


    assert count == n_test_users
    pool.close()
    return result

## testset evaluate
users_to_test = list(testset.keys())
ret = test(model, users_to_test, drop_flag=False)

## learning curve
hist = np.load(args.model_save_path.replace('.pth', '.npy'), allow_pickle=True).item()
hist.keys()

plt.figure(figsize=(6, 4))

epochs = [i+1 for i in range(len(hist['train_loss']))]

plt.plot(epochs, hist['train_loss'], marker='o', label='NGCF', linestyle='-', linewidth=1)
plt.title('Amazon Book Learning History', fontsize=12)
plt.xlabel('Epochs', fontsize=10)
plt.ylabel('train loss value', fontsize=10)

plt.legend(fontsize=8, loc='lower right', frameon=True)

## performance history
plt.figure(figsize=(6, 4))

precision = [hist['val_score'][i]['precision'][0] for i in range(len(hist['val_score']))]
recall = [hist['val_score'][i]['recall'][0] for i in range(len(hist['val_score']))]
ndcg = [hist['val_score'][i]['ndcg'][0] for i in range(len(hist['val_score']))]
hit_ratio = [hist['val_score'][i]['hit_ratio'][0] for i in range(len(hist['val_score']))]

plt.plot(epochs, precision, marker='o', label='precision@20', linestyle='-', linewidth=1)
plt.plot(epochs, recall, marker='^', label='recall@20', linestyle='--', linewidth=1)
plt.plot(epochs, ndcg, marker='s', label='ndcg@20', linestyle='-', linewidth=1)
plt.plot(epochs, hit_ratio, marker='o', label='hr@20', linestyle='--', linewidth=1)

plt.title('Amazon Book Validation Performance', fontsize=12)
plt.xlabel('Epochs', fontsize=10)
plt.ylabel('Score', fontsize=10)
# plt.yticks(np.arange(0.55, 0.8, 0.05), fontsize=9)

plt.legend(fontsize=8, loc='lower right', frameon=True)