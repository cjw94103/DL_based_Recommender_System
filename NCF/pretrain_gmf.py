import argparse
import pandas as pd
import torch

from dataset import SampleGenerator
from models.gmf import GMF
from models.mlp import MLP
from train_func import train

# Define a custom argument type for a list of integers
def list_of_ints(arg):
    return list(map(int, arg.split(',')))

parser = argparse.ArgumentParser()

# dataset
parser.add_argument("--trainset_path", type=str, help="trainset csv path", default="./MovieLens_1M/train_ratings.csv")
parser.add_argument("--valset_path", type=str, help="valset csv path", default="./MovieLens_1M/val_ratings.csv")
parser.add_argument("--testset_path", type=str, help="testset csv path", default="./MovieLens_1M/test_ratings.csv")
parser.add_argument("--negative_sample_path", type=str, help="negatives csv path", default="./MovieLens_1M/negatives.csv")

parser.add_argument("--num_negatives", type=int, help="number of negative sample per 1 positive sample", default=4)
parser.add_argument("--num_workers", type=int, help="num workers of generator", default=0)
parser.add_argument("--batch_size", type=int, help="num of batch size", default=512)

parser.add_argument("--total_user_len", type=int, help="length of total user", default=5418)
parser.add_argument("--total_item_len", type=int, help="length of total item", default=3699)

# gmf model
parser.add_argument("--gmf_latent_dim", type=int, help="gmf model latent dim", default=8)

# model save
parser.add_argument("--model_save_path", type=str, help="your model save path", default="./model_result/gmf/gmf.pth")

## optimizer parameter
parser.add_argument("--weight_decay", type=float, help="weight decay of Optimizer", default=0.)
parser.add_argument("--lr", type=float, help="learning rate", default=1e-3)
parser.add_argument("--epochs", type=int, help="num epochs", default=200)

args = parser.parse_args()

## load sample generator
train_ratings = pd.read_csv(args.trainset_path)
val_ratings = pd.read_csv(args.valset_path)
test_ratings = pd.read_csv(args.testset_path)

negatives = pd.read_csv(args.negative_sample_path)
negatives['negative_items'] = negatives['negative_items'].apply(eval)
negatives['negative_samples'] = negatives['negative_samples'].apply(eval)

sample_generator = SampleGenerator(train_ratings=train_ratings, val_ratings=val_ratings, test_ratings=test_ratings, negatives=negatives,
                                  num_negatives=args.num_negatives, batch_size=args.batch_size, num_workers=args.num_workers)

# load gmf models
model = GMF(num_users=args.total_user_len, 
                num_items=args.total_item_len, 
                latent_dim=args.gmf_latent_dim).to('cuda')

# get optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

# train
train(args, model, sample_generator, optimizer)