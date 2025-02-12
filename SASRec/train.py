import torch
import argparse
import numpy as np

from models import SASRec
from sampler import WarpSampler
from utils import AverageMeter
from tqdm import tqdm
from evaluate_func import evaluate_valid

parser = argparse.ArgumentParser()

# preprocessed dataset path
parser.add_argument("--dataset_path", type=str, help="path yout preprocessed dataset", default="./data/Steam.npy")

# data parameter
parser.add_argument("--batch_size", type=int, help="num of batch size", default=128)
parser.add_argument("--maxlen", type=int, help="maximum sequence length", default=50) # movielens 200, amazon, steam 50
parser.add_argument("--seed", type=int, help="set seed", default=7777)

# model parameter
parser.add_argument("--hidden_units", type=int, help="num of hidden units", default=50)
parser.add_argument("--num_blocks", type=int, help="num of self-attention block", default=2)
parser.add_argument("--num_heads", type=int, help="1 : only self-attention >1 : multi-head attention", default=1)
parser.add_argument("--dropout_rate", type=float, help="dropout rate", default=0.5) # movielens 0.2 amazom, stam 0.5 
parser.add_argument("--l2_emb", type=float, help="weight decay for embedding layer", default=0.0)
parser.add_argument('--device', default='cuda', type=str)

# optimizer parameter
parser.add_argument("--lr", type=float, help="learning rate", default=0.001)
parser.add_argument("--epochs", type=int, help="num epochs", default=1000)

# model save
parser.add_argument("--model_save_path", type=str, help="your model save path", default="./model_result/Steam/Steam.pth")
parser.add_argument("--patience", type=int, help="early stopping", default=20)

args = parser.parse_args()

## load dataset
dataset = np.load(args.dataset_path, allow_pickle=True).item()
user_train, user_valid, user_test, usernum, itemnum = dataset['user_train'], dataset['user_valid'], dataset['user_test'], dataset['usernum'], dataset['itemnum']

## sampler
sampler = WarpSampler(user_train, usernum, itemnum, batch_size=args.batch_size, maxlen=args.maxlen, n_workers=1)

## load model
model = SASRec(usernum, itemnum, args).to(args.device)

for name, param in model.named_parameters():
    try:
        torch.nn.init.xavier_normal_(param.data)
    except:
        pass # just ignore those failed init layers

model.pos_emb.weight.data[0, :] = 0 # for padding index
model.item_emb.weight.data[0, :] = 0 # for padding index

## get optimizer
adam_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))
bce_criterion = torch.nn.BCEWithLogitsLoss()

## start training
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
num_iter = (len(user_train) - 1) // args.batch_size + 1
patience = 0

## training stage
for epoch in range(start_epoch, args.epochs):
    model.train()
    train_losses = AverageMeter()
    train_t = tqdm(range(num_iter))

    for step in train_t:
        # prediction
        u, seq, pos, neg = sampler.next_batch()
        u, seq, pos, neg = np.array(u), np.array(seq), np.array(pos), np.array(neg)
        pos_logits, neg_logits = model(u, seq, pos, neg)
        pos_labels, neg_labels = torch.ones(pos_logits.shape, device=args.device), torch.zeros(neg_logits.shape, device=args.device)

        # calculate loss
        indices = np.where(pos != 0)
        loss = bce_criterion(pos_logits[indices], pos_labels[indices])
        loss += bce_criterion(neg_logits[indices], neg_labels[indices])
        for param in model.item_emb.parameters(): loss += args.l2_emb * torch.norm(param)

        # update weight
        adam_optimizer.zero_grad()
        loss.backward()
        adam_optimizer.step()

        # loss recording
        train_losses.update(loss.item())

        # print tqdm
        print_loss = round(loss.item(), 4)
        train_t.set_postfix_str("Train loss : {}".format(print_loss))

    # record train losses
    train_losses_avg.append(train_losses.avg)

    # validation stage
    model.eval()
    print("validation stage.........")
    ndcg, hit_ratio = evaluate_valid(model, user_train, user_valid, usernum, itemnum, args) # NDCG, HR
    avg_score = (hit_ratio + ndcg) / 2
    val_score_avg.append(avg_score)

    # save best model
    patience += 1 # for early stopping
    if avg_score > minimum_val_score:
        print('improve validation score!! so model save {} -> {}'.format(minimum_val_score, avg_score))
        minimum_val_score = avg_score
        torch.save(model.state_dict(), args.model_save_path)
        patience = 0

    # save history
    save_history(train_losses_avg, val_score_avg, save_path=args.model_save_path.replace('.pth', '.npy'))

    # early stopping
    if patience == args.patience:
        print("early stoppping!!")
        break # stop training