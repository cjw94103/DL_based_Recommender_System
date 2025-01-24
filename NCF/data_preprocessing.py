import pandas as pd
import numpy as np
import argparse
from tqdm import tqdm

from copy import deepcopy
import random

## argparse
parser = argparse.ArgumentParser()

parser.add_argument("--data_path", type=str, help="your dataset path", default="./MovieLens_1M/ratings.dat")

parser.add_argument("--val_sample_size", type=int, help="number of validation data per user", default=1)

parser.add_argument("--save_trainset_path", type=str, help="save path trainset csv", default="./MovieLens_1M/train_ratings.csv")
parser.add_argument("--save_valset_path", type=str, help="save path valset csv", default="./MovieLens_1M/val_ratings.csv")
parser.add_argument("--save_testset_path", type=str, help="save path testset csv", default="./MovieLens_1M/test_ratings.csv")
parser.add_argument("--save_negative_sample_path", type=str, help="save path negative sample csv", default="./MovieLens_1M/negatives.csv")

args = parser.parse_args()

## 필요 함수
def _binarize(ratings):
    """binarize into 0 or 1, imlicit feedback"""
    ratings = deepcopy(ratings)
    ratings.loc[ratings['rating'] > 0, 'rating'] = 1.0 
    return ratings

def _sample_negative(ratings):
    """return all negative items & 100 sampled negative items (논문, user가 interaction 하지 않은 100개의 item 무작위 샘플링, test item을 이 100개의 item 중에서 ranking을 매김"""
    interact_status = ratings.groupby('userId')['itemId'].apply(set).reset_index().rename(
        columns={'itemId': 'interacted_items'})
    interact_status['negative_items'] = interact_status['interacted_items'].apply(lambda x: item_pool - x) 
    interact_status['negative_samples'] = interact_status['negative_items'].apply(lambda x: random.sample(list(x), 99)) 
    return interact_status[['userId', 'negative_items', 'negative_samples']]

def _split_loo(ratings):
    """leave one out train/test split """
    ratings['rank_latest'] = ratings.groupby(['userId'])['timestamp'].rank(method='first', ascending=False)
    test = ratings[ratings['rank_latest'] == 1]
    train = ratings[ratings['rank_latest'] > 1]
    assert train['userId'].nunique() == test['userId'].nunique()
    return train[['userId', 'itemId', 'rating']], test[['userId', 'itemId', 'rating']]

## load data
ml1m_rating = pd.read_csv(args.data_path, sep='::', header=None, names=['uid', 'mid', 'rating', 'timestamp'], engine='python')

## Reindex
user_id = ml1m_rating[['uid']].drop_duplicates().reindex()
user_id['userId'] = np.arange(len(user_id)) 
ml1m_rating = pd.merge(ml1m_rating, user_id, on=['uid'], how='left') 

item_id = ml1m_rating[['mid']].drop_duplicates()
item_id['itemId'] = np.arange(len(item_id)) 
ml1m_rating = pd.merge(ml1m_rating, item_id, on=['mid'], how='left')

ml1m_rating = ml1m_rating[['userId', 'itemId', 'rating', 'timestamp']]

print('Range of userId is [{}, {}]'.format(ml1m_rating.userId.min(), ml1m_rating.userId.max()))
print('Range of itemId is [{}, {}]'.format(ml1m_rating.itemId.min(), ml1m_rating.itemId.max()))

## Negative sampling and train, val, test split
ratings = deepcopy(ml1m_rating)
preprocess_ratings = _binarize(ratings) 

user_pool = set(ratings['userId'].unique())
item_pool = set(ratings['itemId'].unique())

negatives = _sample_negative(ratings)
train_ratings, test_ratings = _split_loo(preprocess_ratings)

sample_size = args.val_sample_size
val_ratings = pd.DataFrame()

for user_id in tqdm(user_pool, total=len(user_pool)):
    condition = (train_ratings['userId'] == user_id)
    _val_ratings = train_ratings[condition].sample(n=sample_size, random_state=0)
    val_ratings = pd.concat([val_ratings, _val_ratings])
    
train_ratings = train_ratings.drop(val_ratings.index)

## save csv
train_ratings.to_csv(args.save_trainset_path, index=False)
val_ratings.to_csv(args.save_valset_path, index=False)
test_ratings.to_csv(args.save_testset_path, index=False)
negatives.to_csv(args.save_negative_sample_path, index=False)