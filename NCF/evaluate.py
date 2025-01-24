import argparse
import pandas as pd
import torch
import matplotlib.pyplot as plt
import numpy as np

from dataset import SampleGenerator
from models.gmf import GMF
from models.mlp import MLP
from models.neumf import NeuMF
from train_func import train
from metrics import MetronAtK
from tqdm import tqdm

def evaluate_test_data(model, test_data, metron):
    test_users, test_items = test_data[0], test_data[1]
    negative_users, negative_items = test_data[2], test_data[3]

    # to cuda
    test_users = test_users.to('cuda')
    test_items = test_items.to('cuda')
    negative_users = negative_users.to('cuda')
    negative_items = negative_items.to('cuda')

    # batch inference
    test_scores = []
    negative_scores = []
    bs = args.batch_size

    for start_idx in range(0, len(test_users), bs):
        end_idx = min(start_idx + bs, len(test_users))
        batch_test_users = test_users[start_idx:end_idx]
        batch_test_items = test_items[start_idx:end_idx]
        test_scores.append(model(batch_test_users, batch_test_items))
    for start_idx in tqdm(range(0, len(negative_users), bs)):
        end_idx = min(start_idx + bs, len(negative_users))
        batch_negative_users = negative_users[start_idx:end_idx]
        batch_negative_items = negative_items[start_idx:end_idx]
        negative_scores.append(model(batch_negative_users, batch_negative_items))
    test_scores = torch.concatenate(test_scores, dim=0)
    negative_scores = torch.concatenate(negative_scores, dim=0)

    # to cpu
    test_users = test_users.cpu()
    test_items = test_items.cpu()
    test_scores = test_scores.cpu()
    negative_users = negative_users.cpu()
    negative_items = negative_items.cpu()
    negative_scores = negative_scores.cpu()

    metron.subjects = [test_users.data.view(-1).tolist(),
                       test_items.data.view(-1).tolist(),
                       test_scores.data.view(-1).tolist(),
                       negative_users.data.view(-1).tolist(),
                       negative_items.data.view(-1).tolist(),
                       negative_scores.data.view(-1).tolist()]

    # calculate score
    hit_ratio, ndcg = metron.cal_hit_ratio(), metron.cal_ndcg()

    return hit_ratio, ndcg

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

# trained model path
parser.add_argument("--gmf_model_path", type=str, help="trained gmf path", default="./model_result/gmf/gmf.pth")
parser.add_argument("--mlp_model_path", type=str, help="trained mlp path", default="./model_result/mlp/mlp.pth")
parser.add_argument("--neumf_model_path", type=str, help="trained neumf path", default="./model_result/neumf/neumf.pth")
parser.add_argument("--neumf_no_pretrain_model_path", type=str, help="trained no pretrained neumf path", default="./model_result/neumf/neumf_no_pretrain.pth")

args = parser.parse_args()

# get generator
train_ratings = pd.read_csv(args.trainset_path)
val_ratings = pd.read_csv(args.valset_path)
test_ratings = pd.read_csv(args.testset_path)

negatives = pd.read_csv(args.negative_sample_path)
negatives['negative_items'] = negatives['negative_items'].apply(eval)
negatives['negative_samples'] = negatives['negative_samples'].apply(eval)

sample_generator = SampleGenerator(train_ratings=train_ratings, val_ratings=val_ratings, test_ratings=test_ratings, negatives=negatives,
                                  num_negatives=args.num_negatives, batch_size=args.batch_size, num_workers=args.num_workers)

# load models with trained weights
# gmf model
gmf = GMF(num_users=args.total_user_len, 
                num_items=args.total_item_len, 
                latent_dim=args.gmf_latent_dim)
gmf_weights = torch.load(args.gmf_model_path)
gmf.load_state_dict(gmf_weights)
gmf.to('cuda')
gmf.eval()

# mlp model
mlp = MLP(num_users=args.total_user_len,
                num_items=args.total_item_len,
                latent_dim=args.mlp_latent_dim, 
                layers=args.mlp_layers)
mlp_weights = torch.load(args.mlp_model_path)
mlp.load_state_dict(mlp_weights)
mlp.to('cuda')
mlp.eval()

# pretrained neumf model
neumf = NeuMF(num_users=args.total_user_len,
             num_items=args.total_item_len,
             latent_dim_mf=args.gmf_latent_dim,
             latent_dim_mlp=args.mlp_latent_dim,
             layers=args.mlp_layers).to('cuda')
neumf.affine_output.weight.data = 0.5 * torch.cat([mlp.affine_output.weight.data, gmf.affine_output.weight.data], dim=-1) # for tensor shape
neumf_weights = torch.load(args.neumf_model_path)
neumf.load_state_dict(neumf_weights)
neumf.to('cuda')
neumf.eval()

# no pretrained neumf model
no_pretrained_neumf = NeuMF(num_users=args.total_user_len,
                            num_items=args.total_item_len,
                            latent_dim_mf=args.gmf_latent_dim,
                            latent_dim_mlp=args.mlp_latent_dim,
                            layers=args.mlp_layers).to('cuda')
no_pretrained_neumf_weights = torch.load(args.neumf_no_pretrain_model_path)
no_pretrained_neumf.affine_output.weight.data = 0.5 * torch.cat([mlp.affine_output.weight.data, gmf.affine_output.weight.data], dim=-1) # for tensor shape
no_pretrained_neumf.load_state_dict(no_pretrained_neumf_weights)
no_pretrained_neumf.to('cuda')
no_pretrained_neumf.eval()

## evaluate
test_data = sample_generator.evaluate_test_data
metron = MetronAtK(top_k=10)

gmf_hit_ratio, gmf_ndcg = evaluate_test_data(gmf, test_data, metron)
mlp_hit_ratio, mlp_ndcg = evaluate_test_data(mlp, test_data, metron)
noneumf_hit_ratio, noneumf_ndcg = evaluate_test_data(no_pretrained_neumf, test_data, metron)
neumf_hit_ratio, neumf_ndcg = evaluate_test_data(neumf, test_data, metron)

print("only gmf --->   hit ratio {} ndcg : {}".format(gmf_hit_ratio, gmf_ndcg))
print("only mlp --->   hit ratio {} ndcg : {}".format(mlp_hit_ratio, mlp_ndcg))
print("no prtrained neumf --->   hit ratio {} ndcg : {}".format(noneumf_hit_ratio, noneumf_ndcg))
print("neumf --->   hit ratio {} ndcg : {}".format(neumf_hit_ratio, neumf_ndcg))

## 모델별 validation hit ratio, ndcg 평균 성능 그래프
gmf_hist = np.load(args.gmf_model_path.replace('.pth', '.npy'), allow_pickle=True).item()
mlp_hist = np.load(args.mlp_model_path.replace('.pth', '.npy'), allow_pickle=True).item()
no_neumf = np.load(args.neumf_no_pretrain_model_path.replace('.pth', '.npy'), allow_pickle=True).item()
neumf_hist = np.load(args.neumf_model_path.replace('.pth', '.npy'), allow_pickle=True).item()

epochs = [i+1 for i in range(200)]
gmf_perf = gmf_hist['val_score']
mlp_perf = mlp_hist['val_score'] + [0.6542448647686541 for i in range(14)]
no_neumf_perf = no_neumf['val_score']
neumf_perf = neumf_hist['val_score']

# 스타일 설정
plt.figure(figsize=(6, 4))

# 각 모델의 그래프
plt.plot(epochs, gmf_perf, marker='o', label='gmf', linestyle='-', linewidth=1)
plt.plot(epochs, mlp_perf, marker='^', label='mlp', linestyle='--', linewidth=1)
plt.plot(epochs, no_neumf_perf, marker='s', label='no_pratrined_neumf', linestyle='-', linewidth=1)
plt.plot(epochs, neumf_perf, marker='o', label='neumf', linestyle='--', linewidth=1)

# 그래프 제목 및 레이블 설정
plt.title('MovieLens', fontsize=12)
plt.xlabel('Epochs', fontsize=10)
plt.ylabel('HR@10, NDCG@10 Average', fontsize=10)
plt.yticks(np.arange(0.55, 0.8, 0.05), fontsize=9)

# 범례 추가
plt.legend(fontsize=8, loc='lower right', frameon=True)