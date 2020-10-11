from collections import defaultdict

from surprise import SVD
from surprise import KNNBasic
from surprise import KNNWithZScore
from surprise import Dataset
from surprise import accuracy
from surprise import Reader
from surprise.model_selection import train_test_split

import os
import pandas as pd


path = '../Datasets/Movielens'
os.chdir(path)
trans = pd.read_csv('u_data.csv')
trans = trans.iloc[:,0:3]
trans.columns = ['user','item','rating']

reader = Reader(rating_scale=(1,5)) 
data = Dataset.load_from_df(trans, reader)
trainset, testset  = train_test_split(data, test_size=0.002)

algo = SVD(n_factors = 50)
algo.fit(trainset)
preds = algo.test(testset)
accuracy.mae(preds)
