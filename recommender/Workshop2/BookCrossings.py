from surprise import SVD
from surprise import KNNBasic
from surprise import KNNWithZScore
from surprise import Dataset
from surprise import accuracy
from surprise import Reader
from surprise.model_selection import train_test_split

import os
import pandas as pd


path = '../Datasets/BookCrossings'
os.chdir(path)
trans = pd.read_csv('BX-Book-Ratings.csv', sep=';', error_bad_lines=False, encoding="latin-1")
trans.columns = ['user','item','rating']
trans = trans[trans.rating != 0]

min_item_ratings = 10
popular_items = trans['item'].value_counts() >= min_item_ratings
popular_items = popular_items[popular_items].index.tolist()

min_user_ratings = 10
active_users = trans['user'].value_counts() >= min_user_ratings
active_users = active_users[active_users].index.tolist()

trans = trans[(trans['item'].isin(popular_items)) & (trans['user'].isin(active_users))] 
reader = Reader(rating_scale=(1,10)) 
data = Dataset.load_from_df(trans, reader)
trainset, testset  = train_test_split(data, test_size=0.002)


algo = SVD(n_factors = 50)

algo.fit(trainset)
preds = algo.test(testset)
accuracy.mae(preds)

