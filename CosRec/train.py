import argparse

import torch.optim as optim

from dataset import Interactions
from utils import *

from models import CosRec, CosRec_base
from tqdm import tqdm
from metrics import evaluate_ranking

## argparse
parser = argparse.ArgumentParser()

# dataset
parser.add_argument("--dataset", type=str, help="ml1m or gowalla", default="gowalla")
parser.add_argument('--data_root', type=str, default='./data/')
parser.add_argument('--train_dir', type=str, default='/test/train.txt')
parser.add_argument('--test_dir', type=str, default='/test/test.txt')
parser.add_argument('--L', type=int, default=5)
parser.add_argument('--T', type=int, default=3)

# train parameter
parser.add_argument('--seed', type=int, default=123)
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--learning_rate', type=float, default=1e-3)
parser.add_argument('--l2', type=float, default=5e-6)
parser.add_argument('--neg_samples', type=int, default=3)
parser.add_argument('--use_cuda', type=str2bool, default=True)
parser.add_argument('--epochs', type=int, default=40)

# model parameter
parser.add_argument('--model_type', type=str, default='cnn')
parser.add_argument('--d', type=int, default=50)
parser.add_argument('--block_num', type=int, default=2, help='number of CNN blocks')
parser.add_argument('--block_dim', type=list, default=[128, 256])
parser.add_argument('--drop', type=float, default=0.5, help='drop out ratio.')
parser.add_argument('--fc_dim', type=int, default=150)
parser.add_argument('--ac_fc', type=str, help="relu or tanh or sigm", default='tanh')

# model save
parser.add_argument("--model_save_path", type=str, help="your model save path", default="./model_result/gowalla/CosRec_CNN.pth")

args = parser.parse_args()

## set seed
set_seed(args.seed, cuda=args.use_cuda)

## load dataset
train = Interactions(args.data_root+args.dataset+args.train_dir)
train.to_sequence(args.L, args.T)

test = Interactions(args.data_root+args.dataset+args.test_dir,
                        user_map=train.user_map,
                        item_map=train.item_map)
test_sequence = train.test_sequences

## load model
_num_items = train.num_items
_num_users = train.num_users
_device = torch.device("cuda" if args.use_cuda else "cpu")

if args.model_type == 'mlp':
    model = CosRec_base(_num_users, _num_items, args.L, args.d).to(_device)
elif args.model_type == 'cnn':
    model = CosRec(_num_users, _num_items, args.L, 
                args.d, block_num=args.block_num, block_dim=args.block_dim, 
                fc_dim = args.fc_dim, ac_fc = args.ac_fc,
                drop_prob=args.drop).to(_device)

## get optimizer
_optimizer = optim.Adam(model.parameters(), weight_decay=args.l2, lr=args.learning_rate)

## training
_candidate = {}

def _generate_negative_samples(users, interactions, n):
    """
    Sample negative from a candidate set of each user. The
    candidate set of each user is defined by:
    {All Items} \ {Items Rated by User}

    Parameters
    ----------

    users: array of np.int64
        sequence users
    interactions: :class:`spotlight.interactions.Interactions`
        training instances, used for generate candidates
    n: int
        total number of negatives to sample for each sequence
    """

    users_ = users.squeeze()
    negative_samples = np.zeros((users_.shape[0], n), np.int64)
    if not _candidate:
        all_items = np.arange(interactions.num_items - 1) + 1  # 0 for padding
        train = interactions.tocsr()
        for user, row in enumerate(train):
            _candidate[user] = list(set(all_items) - set(row.indices))

    for i, u in enumerate(users_):
        for j in range(n):
            x = _candidate[u]
            negative_samples[i, j] = x[
                np.random.randint(len(x))]

    return negative_samples

def save_history(train_loss_list, val_score_list, save_path):
    history = {}

    history['train_loss'] = train_loss_list
    history['val_score'] = val_score_list

    np.save(save_path, history)

start_epoch = 0
global_step = 0

train_losses_avg = []
val_score_avg = []
minimum_val_score = 0
num_iter = (len(train) - 1) // args.batch_size + 1
patience = 0

# convert to sequences, targets and users
sequences_np = train.sequences.sequences
targets_np = train.sequences.targets
users_np = train.sequences.user_ids.reshape(-1, 1)

L, T = train.sequences.L, train.sequences.T

for epoch in range(start_epoch, args.epochs):
    model.train()
    train_losses = AverageMeter()

    users_np, sequences_np, targets_np = shuffle(users_np, sequences_np, targets_np)

    negatives_np = _generate_negative_samples(users_np, train, n=args.neg_samples)

    # convert numpy arrays to PyTorch tensors and move it to the corresponding devices
    users, sequences, targets, negatives = (torch.from_numpy(users_np).long(),
                                            torch.from_numpy(sequences_np).long(),
                                            torch.from_numpy(targets_np).long(),
                                            torch.from_numpy(negatives_np).long())

    users, sequences, targets, negatives = (users.to(_device),
                                            sequences.to(_device),
                                            targets.to(_device),
                                            negatives.to(_device))

    train_t = tqdm(enumerate(minibatch(users, sequences, targets, negatives, batch_size=args.batch_size)), total=num_iter)

    for (minibatch_num, (batch_users, batch_sequences, batch_targets, batch_negatives)) in train_t:
        # predict
        items_to_predict = torch.cat((batch_targets, batch_negatives), 1)
        items_prediction = model(batch_sequences, batch_users, items_to_predict)
        (targets_prediction, negatives_prediction) = torch.split(items_prediction, [batch_targets.size(1), batch_negatives.size(1)], dim=1)

        # calculate loss
        positive_loss = -torch.mean( torch.log(torch.sigmoid(targets_prediction)))
        negative_loss = -torch.mean(torch.log(1 - torch.sigmoid(negatives_prediction)))
        loss = positive_loss + negative_loss

        # update weights
        _optimizer.zero_grad()
        loss.backward()
        _optimizer.step()

        # train loss recording
        train_losses.update(loss.item())

        # print tqdm
        print_loss = round(loss.item(), 4)
        train_t.set_postfix_str("Train loss : {}".format(print_loss))

    # record train loss
    train_losses_avg.append(train_losses.avg)

    # validation stage
    precision, recall, mean_aps = evaluate_ranking(model, test, test_sequence, train.num_items, train.num_users, _device, train, k=[1, 5, 10])
    avg_score = float(mean_aps)
    
    val_perf_dict = {}
    val_perf_dict['precision_1'] = float(precision[0].mean())
    val_perf_dict['precision_5'] = float(precision[1].mean())
    val_perf_dict['precision_10'] = float(precision[2].mean())
    val_perf_dict['recall_1'] = float(recall[0].mean())
    val_perf_dict['recall_5'] = float(recall[1].mean())
    val_perf_dict['recall_10'] = float(recall[2].mean())

    val_score_avg.append(val_perf_dict)

    # save history
    save_history(train_losses_avg, val_score_avg, save_path=args.model_save_path.replace('.pth', '.npy'))

    # save best model
    if avg_score > minimum_val_score:
        print('improve validation score!! so model save {} -> {}'.format(minimum_val_score, avg_score))
        minimum_val_score = avg_score
        torch.save(model.state_dict(), args.model_save_path)