import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt

from models import SASRec
from sampler import WarpSampler
from utils import AverageMeter
from tqdm import tqdm
from evaluate_func import evaluate

## argparse
parser = argparse.ArgumentParser()

# preprocessed dataset path
parser.add_argument("--dataset_path", type=str, help="path yout preprocessed dataset", default="./data/Video.npy")

# data parameter
parser.add_argument("--batch_size", type=int, help="num of batch size", default=128)
parser.add_argument("--maxlen", type=int, help="maximum sequence length", default=50) # movielens 200, amazon, steam 50
parser.add_argument("--seed", type=int, help="set seed", default=7777)

# model parameter
parser.add_argument("--hidden_units", type=int, help="num of hidden units", default=50)
parser.add_argument("--num_blocks", type=int, help="num of self-attention block", default=2)
parser.add_argument("--num_heads", type=int, help="1 : only self-attention >1 : multi-head attention", default=1)
parser.add_argument('--device', default='cuda', type=str)

# optimizer parameter
parser.add_argument("--lr", type=float, help="learning rate", default=0.001)
parser.add_argument("--epochs", type=int, help="num epochs", default=1000)

# model save
parser.add_argument("--model_save_path", type=str, help="your model save path", default="./model_result/Video/Video.pth")

args = parser.parse_args()
args.dropout_rate = 0.

## load dataset
dataset = np.load(args.dataset_path, allow_pickle=True).item()
user_train, user_valid, user_test, usernum, itemnum = dataset['user_train'], dataset['user_valid'], dataset['user_test'], dataset['usernum'], dataset['itemnum']

## load model
weights = torch.load(args.model_save_path)

model = SASRec(usernum, itemnum, args)
model.load_state_dict(weights)
model.to(args.device)
model.eval()

## testset performance
ndcg, hit_ratio = evaluate(model, user_train, user_valid, user_test, usernum, itemnum, args)
print(hit_ratio, ndcg)