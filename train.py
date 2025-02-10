import torch
import torch.optim as optim
import argparse
import scipy.sparse as sp
import numpy as np
import metrics
import multiprocessing
import heapq

from models import NGCF
from dataset import SampleGenerator
from tqdm import tqdm
from utils import AverageMeter

## save history function
def save_history(train_loss_list, val_score_list, save_path):
    history = {}

    history['train_loss'] = train_loss_list
    history['val_score'] = val_score_list

    np.save(save_path, history)

## argparse
# Define a custom argument type for a list of integers
def list_of_ints(arg):
    return list(map(int, arg.split(',')))

def list_of_floats(arg):
    return list(map(float, arg.split(',')))

parser = argparse.ArgumentParser()

# path preprocessed dataset
parser.add_argument("--trainset_path", type=str, help="save path trainset npy", default="./data/gowalla/trainset.npy")
parser.add_argument("--valset_path", type=str, help="save path valset npy", default="./data/gowalla/valset.npy")
parser.add_argument("--testset_path", type=str, help="save path testset npy", default="./data/gowalla/testset.npy")

parser.add_argument("--user_item_info_path", type=str, help="save path user_item_info npy", default="./data/gowalla/user_item_info.npy")

parser.add_argument("--adj_mat_path", type=str, help="save path adj_mat npz", default="./data/gowalla/s_adj_mat.npz")
parser.add_argument("--norm_adj_mat_path", type=str, help="save path norm_adj_mat npz", default="./data/gowalla/s_norm_adj_mat.npz")
parser.add_argument("--mean_mat_path", type=str, help="save path mean_adj_mat npz", default="./data/gowalla/s_mean_adj_mat.npz")

# gpu parameter
parser.add_argument("--gpu_id", type=str, help="your gpu id", default="0")

# model parameter
parser.add_argument("--batch_size", type=int, help="num of batch size", default=512)
parser.add_argument("--embed_size", type=int, help="number of embedding size for user and item", default=64)
parser.add_argument("--node_dropout", type=float, help="node dropout ratio", default=0.1)
parser.add_argument("--mess_dropout", type=list_of_floats, help="message dropout ratio each layer", default="0.1, 0.1, 0.1")
parser.add_argument("--layer_size", type=list_of_ints, help="output size of embed prop. layer", default="64, 64, 64")
parser.add_argument("--weight_decay", type=float, help="l2 decay", default=1e-5)

# model save
parser.add_argument("--model_save_path", type=str, help="your model save path", default="./model_result/NGCF_Gowalla.pth")

# optimizer parameter
parser.add_argument("--lr", type=float, help="learning rate", default=0.0005)
parser.add_argument("--epochs", type=int, help="num epochs", default=200)

args = parser.parse_args()

# get data generator
data_generator = SampleGenerator(trainset_path=args.trainset_path, 
                                 valset_path=args.valset_path,
                                 user_item_info_path=args.user_item_info_path, 
                                 batch_size=args.batch_size)

# define device
device = torch.device('cuda:' + str(args.gpu_id))

# load adj mat
norm_adj = sp.load_npz(args.norm_adj_mat_path)

# 학습할 때 dropout flag=True
dropout_flag = True

# load model
model = NGCF(n_user=data_generator.n_users,
            n_item=data_generator.n_items,
            norm_adj=norm_adj,
            device=device,
            embed_size=args.embed_size,
            batch_size=args.batch_size,
            node_dropout=args.node_dropout,
            mess_dropout=args.mess_dropout,
            layer_size=args.layer_size,
            weight_decay=args.weight_decay).to(device)

# get optimizer
optimizer = optim.Adam(model.parameters(), lr=args.lr)

# Start Training
start_epoch = 0
global_step = 0
minimum_val_score = 0

train_losses_avg = []
val_score_avg = []

val_data = data_generator.valset

USR_NUM, ITEM_NUM = data_generator.n_users, data_generator.n_items
N_TRAIN, N_TEST = data_generator.n_train, data_generator.n_val
BATCH_SIZE = args.batch_size
Ks = [20]

# for parallel validation
cores = multiprocessing.cpu_count() // 2

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
    user_pos_test = data_generator.valset[u]

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

## training stage
for epoch in range(start_epoch, args.epochs):
    train_losses = AverageMeter()
    train_iters = data_generator.n_train // args.batch_size + 1
    train_t = tqdm(range(train_iters))

    for idx in train_t:
        users, pos_items, neg_items = data_generator.sample()
        u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings = model(users,
                                                                       pos_items,
                                                                       neg_items,
                                                                       drop_flag=True)
        
        # calculate loss
        batch_loss, batch_mf_loss, batch_emb_loss = model.create_bpr_loss(u_g_embeddings,
                                                                          pos_i_g_embeddings,
                                                                          neg_i_g_embeddings)
        train_losses.update(batch_loss.item())

        # update weights
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

        # print tqdm
        print_loss = round(batch_loss.item(), 4)
        train_t.set_postfix_str("Train loss : {}".format(print_loss))

    # record train losses
    train_losses_avg.append(train_losses.avg)

    # validation stage
    users_to_val = list(data_generator.valset.keys())
    ret = test(model, users_to_val, drop_flag=False)
    val_recall = ret['recall'][0]
    val_score_avg.append(ret)

    # save best model (recall@20 기준)
    if val_recall > minimum_val_score:
        print('improve validation score!! so model save {} -> {}'.format(minimum_val_score, val_recall))
        minimum_val_score = val_recall
        torch.save(model.state_dict(), args.model_save_path)

    # save history
    save_history(train_losses_avg, val_score_avg, save_path=args.model_save_path.replace('.pth', '.npy'))

    # save per epoch model
    torch.save(model.state_dict(), args.model_save_path.replace('.pth', '_per_epoch.pth'))