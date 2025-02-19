import warnings
warnings.filterwarnings('ignore')

import numpy as np
import argparse
import torch
import torch.nn.functional as F
import metrics

from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from utils import collate_fn, AverageMeter
from models import NARM
from dataset import load_data, RecSysDataset

## argparse
parser = argparse.ArgumentParser()

# dataset root path
parser.add_argument("--dataset_path", type=str, help="root dataset path", default="./data/yoochoose1_4/")
parser.add_argument('--n_items', type=int, help='number of items, diginetica 43098, yoochooise 37484', default=37484)

# data parameter
parser.add_argument("--topk", type=int, help="number of top score items selected for calculating recall and mrr metrics", default=20)
parser.add_argument('--valid_portion', type=float, help='split the portion of training set as validation set', default=0.1)

# model parameter
parser.add_argument('--hidden_size', type=int, help='hidden state size of gru module', default=100)
parser.add_argument('--embed_dim', type=int, help='the dimension of item embedding', default=50)

# learning parameter
parser.add_argument("--batch_size", type=int, help="num of batch size", default=512)
parser.add_argument("--epochs", type=int, help="num epochs", default=200)
parser.add_argument("--lr", type=float, help="learning rate", default=0.0005)
parser.add_argument("--lr_decay_rate", type=float, help="learning rate decay rate", default=0.1)
parser.add_argument("--lr_decay_step", type=int, help="the number of steps after which the learning rate decay", default=160)

# model save
parser.add_argument("--model_save_path", type=str, help="your model save path", default="./model_result/NARM_yoochoose1_4.pth")

args = parser.parse_args()

## get dataloader
train, valid, test = load_data(args.dataset_path, valid_portion=args.valid_portion)

train_data = RecSysDataset(train)
valid_data = RecSysDataset(valid)

train_loader = DataLoader(train_data, batch_size = args.batch_size, shuffle = True, collate_fn = collate_fn)
valid_loader = DataLoader(valid_data, batch_size = args.batch_size, shuffle = False, collate_fn = collate_fn)

## load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NARM(args.n_items, args.hidden_size, args.embed_dim, args.batch_size).to(device)

## get optimizer & criterion
optimizer = torch.optim.Adam(model.parameters(), args.lr)
criterion = torch.nn.CrossEntropyLoss()
scheduler = StepLR(optimizer, step_size = args.lr_decay_step, gamma = args.lr_decay_rate)

## start training!!
def save_history(train_loss_list, val_score_list, save_path):
    history = {}

    history['train_loss'] = train_loss_list
    history['val_score'] = val_score_list

    np.save(save_path, history)

start_epoch = 0

train_losses_avg = []
val_score_avg = []
minimum_val_score = 0


for epoch in range(start_epoch, args.epochs):
    # training stage
    model.train()
    train_losses = AverageMeter()
    train_t = tqdm(enumerate(train_loader), total=len(train_loader))

    # lr schedule
    scheduler.step(epoch = epoch)

    for i, (seq, target, lens) in train_t:
        seq = seq.to(device) # seq는 zero-padding 수행, NARM 클래스에서 pad_packed_sequence 함수를 통해 GRU가 Padding 부분은 학습에서 무시
        target = target.to(device)
        outputs = model(seq, lens)

        # calculate loss
        loss = criterion(outputs, target)

        # update weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step() 

        # loss recording
        train_losses.update(loss.item())

        # print tqdm
        print_loss = round(loss.item(), 4)
        train_t.set_postfix_str("Train loss : {}".format(print_loss))
        
    # record train losses
    train_losses_avg.append(train_losses.avg)

    # validation stage
    model.eval()
    val_t = tqdm(enumerate(valid_loader), total=len(valid_loader))

    with torch.no_grad():
        recall_score = AverageMeter()
        mrr_score = AverageMeter()
        # val_score_list = []
        for i, (seq, target, lens) in val_t:
            seq = seq.to(device)
            target = target.to(device)
            outputs = model(seq, lens)
            logits = F.softmax(outputs, dim = 1)
            
            recall, mrr = metrics.evaluate(logits, target, k = args.topk)
            
            recall_score.update(recall)
            mrr_score.update(mrr)
            # val_score_list.append((recall + mrr)/2)

    # avg_score = np.array(val_score_list).mean()
    avg_score = (recall_score.avg + mrr_score.avg) / 2
    
    val_perf_dict = {}
    val_perf_dict['recall'] = recall_score.avg
    val_perf_dict['mrr'] = mrr_score.avg
    val_score_avg.append(val_perf_dict)

    # save best model
    if avg_score > minimum_val_score:
        print('improve validation score!! so model save {} -> {}'.format(minimum_val_score, avg_score))
        minimum_val_score = avg_score
        torch.save(model.state_dict(), args.model_save_path)

    # save history
    save_history(train_losses_avg, val_score_avg, save_path=args.model_save_path.replace('.pth', '.npy'))