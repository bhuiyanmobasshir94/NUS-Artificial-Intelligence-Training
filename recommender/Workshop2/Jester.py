from surprise import SVD
from surprise import KNNBasic
from surprise import KNNWithZScore
from surprise import Dataset
from surprise import accuracy
from surprise import Reader
from surprise.model_selection import train_test_split

import os
import pandas as pd

path = "../Datasets/Jester"
os.chdir(path)
trans = pd.read_csv("jester_ratings.dat", sep='\s+', names=['user', 'item', 'rating'])

trans = trans[trans.rating != 99]

print(trans.shape)
min_item_ratings = 20000
popular_items = trans['item'].value_counts() >= min_item_ratings
popular_items = popular_items[popular_items].index.tolist()

min_user_ratings = 100
active_users = trans['user'].value_counts() >= min_user_ratings
active_users = active_users[active_users].index.tolist()

trans = trans[(trans['item'].isin(popular_items)) & (trans['user'].isin(active_users))] 
print(trans.shape)

reader = Reader(rating_scale=(-10,10)) 
data = Dataset.load_from_df(trans, reader)
trainset, testset  = train_test_split(data, test_size=0.002); 

algo = SVD(n_factors = 50)
algo.fit(trainset)
preds = algo.test(testset)
accuracy.mae(preds)