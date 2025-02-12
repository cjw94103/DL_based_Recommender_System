import numpy as np
import argparse
from collections import defaultdict

## 필요 함수
def build_index(data_path):

    ui_mat = np.loadtxt(data_path, dtype=np.int32)

    n_users = ui_mat[:, 0].max()
    n_items = ui_mat[:, 1].max()

    u2i_index = [[] for _ in range(n_users + 1)]
    i2u_index = [[] for _ in range(n_items + 1)]

    for ui_pair in ui_mat:
        u2i_index[ui_pair[0]].append(ui_pair[1])
        i2u_index[ui_pair[1]].append(ui_pair[0])

    return u2i_index, i2u_index

def data_partition(data_path):
    usernum = 0
    itemnum = 0
    User = defaultdict(list)
    user_train = {}
    user_valid = {}
    user_test = {}
    # assume user/item index starting from 1
    f = open(data_path, 'r')
    for line in f:
        u, i = line.rstrip().split(' ')
        u = int(u)
        i = int(i)
        usernum = max(u, usernum)
        itemnum = max(i, itemnum)
        User[u].append(i)

    for user in User:
        nfeedback = len(User[user])
        if nfeedback < 3:
            user_train[user] = User[user]
            user_valid[user] = []
            user_test[user] = []
        else:
            user_train[user] = User[user][:-2]
            user_valid[user] = []
            user_valid[user].append(User[user][-2])
            user_test[user] = []
            user_test[user].append(User[user][-1])
    return [user_train, user_valid, user_test, usernum, itemnum]

## argparse
parser = argparse.ArgumentParser()

# raw data path
parser.add_argument("--data_path", type=str, help="your raw data path (.txt)", default="./data/ml-1m.txt")

# save path
parser.add_argument("--save_dataset_path", type=str, help="save path preprocessed dataset", default="./data/ml_1m.npy")

args = parser.parse_args()

## preprocess
u2i_index, i2u_index = build_index(args.data_path)
dataset = data_partition(args.data_path)
user_train, user_valid, user_test, usernum, itemnum = dataset

cc = 0.0
for u in user_train:
    cc += len(user_train[u])
print('average sequence length: %.2f' % (cc / len(user_train)))

## save dataset
save_dataset = {}

save_dataset['u2i_index'] = u2i_index
save_dataset['i2u_index'] = i2u_index

save_dataset['user_train'] = user_train
save_dataset['user_valid'] = user_valid
save_dataset['user_test'] = user_test
save_dataset['usernum'] = usernum
save_dataset['itemnum'] = itemnum

np.save(args.save_dataset_path, save_dataset)