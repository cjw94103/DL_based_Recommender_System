import argparse
import matplotlib.pyplot as plt

from dataset import Interactions
from utils import *

from models import CosRec, CosRec_base
from tqdm import tqdm
from metrics import evaluate_ranking

## argparse
parser = argparse.ArgumentParser()

# dataset
parser.add_argument("--dataset", type=str, help="ml1m or gowalla", default="m1lm")
parser.add_argument('--data_root', type=str, default='./data/')
parser.add_argument('--train_dir', type=str, default='/test/train.txt')
parser.add_argument('--test_dir', type=str, default='/test/test.txt')
parser.add_argument('--L', type=int, default=5)
parser.add_argument('--T', type=int, default=3)

# train parameter
parser.add_argument('--seed', type=int, default=123)
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--use_cuda', type=str2bool, default=True)

# model parameter
parser.add_argument('--model_type', type=str, default='cnn')
parser.add_argument('--d', type=int, default=50)
parser.add_argument('--block_num', type=int, default=2, help='number of CNN blocks')
parser.add_argument('--block_dim', type=list, default=[128, 256])
parser.add_argument('--drop', type=float, default=0.5, help='drop out ratio.')
parser.add_argument('--fc_dim', type=int, default=150)
parser.add_argument('--ac_fc', type=str, help="relu or tanh or sigm", default='tanh')

# model save
parser.add_argument("--model_save_path", type=str, help="your model save path", default="./model_result/m1lm/CosRec_CNN.pth")

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

## load trained model
_num_items = train.num_items
_num_users = train.num_users
_device = torch.device("cuda" if args.use_cuda else "cpu")
weights = torch.load(args.model_save_path)

if args.model_type == 'mlp':
    model = CosRec_base(_num_users, _num_items, args.L, args.d)
elif args.model_type == 'cnn':
    model = CosRec(_num_users, _num_items, args.L, 
                args.d, block_num=args.block_num, block_dim=args.block_dim, 
                fc_dim = args.fc_dim, ac_fc = args.ac_fc,
                drop_prob=args.drop)

model.load_state_dict(weights)
model.eval()
model.to(_device)

## evaluate
precision, recall, mean_aps = evaluate_ranking(model, test, test_sequence, train.num_items, train.num_users, _device, train, k=[1, 5, 10])

val_perf_dict = {}
val_perf_dict['precision_1'] = float(precision[0].mean())
val_perf_dict['precision_5'] = float(precision[1].mean())
val_perf_dict['precision_10'] = float(precision[2].mean())
val_perf_dict['recall_1'] = float(recall[0].mean())
val_perf_dict['recall_5'] = float(recall[1].mean())
val_perf_dict['recall_10'] = float(recall[2].mean())

print(val_perf_dict, float(mean_aps))