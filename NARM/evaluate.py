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
parser.add_argument("--batch_size", type=int, help="num of batch size", default=512)

# model save
parser.add_argument("--model_save_path", type=str, help="your model save path", default="./model_result/NARM_yoochoose1_4.pth")

args = parser.parse_args()

## get test dataloader
train, valid, test = load_data(args.dataset_path, valid_portion=args.valid_portion)
test_data = RecSysDataset(test)
test_loader = DataLoader(test_data, batch_size = args.batch_size, shuffle = False, collate_fn = collate_fn)

## load trained model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
weights = torch.load(args.model_save_path)

model = NARM(args.n_items, args.hidden_size, args.embed_dim, args.batch_size)
model.load_state_dict(weights)
model.to(device)
model.eval()

## evaluate
test_t = tqdm(enumerate(test_loader), total=len(test_loader))

with torch.no_grad():
    recall_score = AverageMeter()
    mrr_score = AverageMeter()

    for i, (seq, target, lens) in test_t:
        seq = seq.to(device)
        target = target.to(device)
        outputs = model(seq, lens)
        logits = F.softmax(outputs, dim = 1)
        
        recall, mrr = metrics.evaluate(logits, target, k = args.topk)
        
        recall_score.update(recall)
        mrr_score.update(mrr)

print("recall@20 : {}, mrr@20 : {}".format(recall_score.avg, mrr_score.avg))