import argparse
import pandas as pd
import torch

from dataset import SampleGenerator
from models.gmf import GMF
from models.mlp import MLP
from models.neumf import NeuMF
from train_func import train

## argparse
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
parser.add_argument("--batch_size", type=int, help="num of batch size", default=1024)

parser.add_argument("--total_user_len", type=int, help="length of total user", default=5418)
parser.add_argument("--total_item_len", type=int, help="length of total item", default=3699)

# gmf model
parser.add_argument("--gmf_latent_dim", type=int, help="gmf model latent dim", default=8)

# mlp model
parser.add_argument("--mlp_latent_dim", type=int, help="mlp model latent dim", default=8)
parser.add_argument("--mlp_layers", type=list_of_ints, help="mlp model layers", default="16, 64, 32, 16, 8")

# pretrained model path
parser.add_argument("--gmf_model_path", type=str, help="pretrained gmf path", default="./model_result/gmf/gmf.pth")
parser.add_argument("--mlp_model_path", type=str, help="pretrained mlp path", default="./model_result/mlp/mlp.pth")

# model save
parser.add_argument("--model_save_path", type=str, help="your model save path", default="./model_result/neumf/neumf_no_pretrain.pth")

## optimizer parameter
parser.add_argument("--weight_decay", type=float, help="weight decay of Optimizer", default=0.0000001)
parser.add_argument("--lr", type=float, help="learning rate", default=1e-3)
parser.add_argument("--epochs", type=int, help="num epochs", default=200)

args = parser.parse_args()

## load SampleGenerator
train_ratings = pd.read_csv(args.trainset_path)
val_ratings = pd.read_csv(args.valset_path)
test_ratings = pd.read_csv(args.testset_path)

negatives = pd.read_csv(args.negative_sample_path)
negatives['negative_items'] = negatives['negative_items'].apply(eval)
negatives['negative_samples'] = negatives['negative_samples'].apply(eval)

sample_generator = SampleGenerator(train_ratings=train_ratings, val_ratings=val_ratings, test_ratings=test_ratings, negatives=negatives,
                                  num_negatives=args.num_negatives, batch_size=args.batch_size, num_workers=args.num_workers)

## load models
model = NeuMF(num_users=args.total_user_len,
             num_items=args.total_item_len,
             latent_dim_mf=args.gmf_latent_dim,
             latent_dim_mlp=args.mlp_latent_dim,
             layers=args.mlp_layers).to('cuda')

## transfer pretrained weights
# gmf model
gmf = GMF(num_users=args.total_user_len, 
                num_items=args.total_item_len, 
                latent_dim=args.gmf_latent_dim)
gmf_weights = torch.load(args.gmf_model_path)
gmf.load_state_dict(gmf_weights)
gmf.to('cuda')

# mlp model
mlp = MLP(num_users=args.total_user_len,
                num_items=args.total_item_len,
                latent_dim=args.mlp_latent_dim, 
                layers=args.mlp_layers)
mlp_weights = torch.load(args.mlp_model_path)
mlp.load_state_dict(mlp_weights)
mlp.to('cuda')

# # weight transfer
# model.embedding_user_mlp.weight.data = mlp.embedding_user.weight.data
# model.embedding_item_mlp.weight.data = mlp.embedding_item.weight.data
# model.embedding_user_mf.weight.data = gmf.embedding_user.weight.data
# model.embedding_item_mf.weight.data = gmf.embedding_item.weight.data
model.affine_output.weight.data = 0.5 * torch.cat([mlp.affine_output.weight.data, gmf.affine_output.weight.data], dim=-1)
torch.nn.init.normal_(model.affine_output.weight.data, 0.0, 0.01)
# model.affine_output.bias.data = 0.5 * (mlp.affine_output.bias.data + gmf.affine_output.bias.data)

## get optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

## train
train(args, model, sample_generator, optimizer)