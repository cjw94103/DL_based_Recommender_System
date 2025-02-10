import numpy as np
import random as rd
import scipy.sparse as sp
import argparse

# set seed
rd.seed(7777)
np.random.seed(7777)

## 필요 함수
def print_statistics(n_users, n_items, n_train, n_test):
    print('n_users=%d, n_items=%d' % (n_users, n_items))
    print('n_interactions=%d' % (n_train + n_test))
    print('n_train=%d, n_test=%d, sparsity=%.5f' % (n_train, n_test, (n_train + n_test)/(n_users * n_items)))

def create_adj_mat(n_users, n_items, R):
    adj_mat = sp.dok_matrix((n_users + n_items, n_users + n_items), dtype=np.float32)
    adj_mat = adj_mat.tolil()
    R = R.tolil()

    adj_mat[:n_users, n_users:] = R
    adj_mat[n_users:, :n_users] = R.T
    adj_mat = adj_mat.todok()

    return adj_mat

def mean_adj_single(adj):
    # D^-1 * A
    rowsum = np.array(adj.sum(1))

    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_inv = sp.diags(d_inv)

    norm_adj = d_mat_inv.dot(adj)
    # norm_adj = adj.dot(d_mat_inv)
    print('generate single-normalized adjacency matrix.')
    
    return norm_adj.tocoo()

## argparse
parser = argparse.ArgumentParser()

parser.add_argument("--trainfile_path", type=str, help="your train data path", default="./data/gowalla/train.txt")
parser.add_argument("--testfile_path", type=str, help="your test data path", default="./data/gowalla/test.txt")
parser.add_argument("--validation_sampling_ratio", type=float, help="number of validation data per user", default=0.1)

parser.add_argument("--save_trainset_path", type=str, help="save path trainset npy", default="./data/gowalla/trainset.npy")
parser.add_argument("--save_valset_path", type=str, help="save path valset npy", default="./data/gowalla/valset.npy")
parser.add_argument("--save_testset_path", type=str, help="save path testset npy", default="./data/gowalla/testset.npy")

parser.add_argument("--save_user_item_info_path", type=str, help="save path user_item_info npy", default="./data/gowalla/user_item_info.npy")

parser.add_argument("--save_adj_mat_path", type=str, help="save path adj_mat npz", default="./data/gowalla/s_adj_mat.npz")
parser.add_argument("--save_norm_adj_mat_path", type=str, help="save path norm_adj_mat npz", default="./data/gowalla/s_norm_adj_mat.npz")
parser.add_argument("--save_mean_mat_path", type=str, help="save path mean_adj_mat npz", default="./data/gowalla/s_mean_adj_mat.npz")

args = parser.parse_args()

train_file = args.trainfile_path
test_file = args.testfile_path

# user_item_info
n_users, n_items = 0, 0
n_train, n_val, n_test = 0, 0, 0
exist_users = []

# trainfile 처리
with open(train_file) as f:
    for l in f.readlines():
        if len(l) > 0:
            l = l.strip('\n').split(' ')
            items = [int(i) for i in l[1:]]
            uid = int(l[0])
            exist_users.append(uid)
            n_items = max(n_items, max(items))
            n_users = max(n_users, uid)
            n_train += len(items)

# testfile 처리
with open(test_file) as f:
    for l in f.readlines():
        if len(l) > 0:
            l = l.strip('\n')
            try:
                items = [int(i) for i in l.split(' ')[1:]]
            except Exception:
                continue
            n_items = max(n_items, max(items))
            n_test += len(items)

n_items += 1
n_users += 1

# make train, val, test user-item interaction
R = sp.dok_matrix((n_users, n_items), dtype=np.float32)

train_items_dict, val_set, test_set = {}, {}, {}
with open(train_file) as f_train:
    with open(test_file) as f_test:
        for l in f_train.readlines():
            if len(l) == 0:
                break
            l = l.strip('\n')
            items = [int(i) for i in l.split(' ')]
            uid, train_items = items[0], items[1:]
            
            # 10% validation set sampling
            val_samp_idx = np.random.choice(len(train_items), int(len(train_items) * args.validation_sampling_ratio), replace=False)
            val_items = [train_items[val_samp_idx[i]] for i in range(len(val_samp_idx))]
            train_items = list(set(train_items) - set(val_items))
            
            for i in train_items:
                R[uid, i] = 1.

            train_items_dict[uid] = train_items
            val_set[uid] = val_items

        for l in f_test.readlines():
            if len(l) == 0: break
            l = l.strip('\n')
            try:
                items = [int(i) for i in l.split(' ')]
            except Exception:
                continue

            uid, test_items = items[0], items[1:]
            test_set[uid] = test_items

# get adj mat
adj_mat = create_adj_mat(n_users, n_items, R).tocsr()
norm_adj_mat = mean_adj_single(adj_mat + sp.eye(adj_mat.shape[0])).tocsr()
mean_adj_mat = mean_adj_single(adj_mat).tocsr()

## save
# user item info
user_item_info = {}
user_item_info['n_users'] = n_users
user_item_info['n_items'] = n_items
user_item_info['n_train'] = n_train
user_item_info['n_val'] = n_val
user_item_info['n_test'] = n_test
user_item_info['exist_users'] = exist_users

np.save(args.save_user_item_info_path, user_item_info)

# dataset
np.save(args.save_trainset_path, train_items_dict)
np.save(args.save_valset_path, val_set)
np.save(args.save_testset_path, test_set)

# adj_mat
sp.save_npz(args.save_adj_mat_path, adj_mat)
sp.save_npz(args.save_norm_adj_mat_path, norm_adj_mat)
sp.save_npz(args.save_mean_mat_path, mean_adj_mat)