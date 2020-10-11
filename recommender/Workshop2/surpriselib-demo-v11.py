# -*- coding: utf-8 -*-
"""
@author: issbas
Demo the Surprise package
"""

from surprise import KNNBasic
from surprise import KNNWithMeans
from surprise import SVD
from surprise import SVDpp
from surprise import NMF
from surprise import Dataset
from surprise import accuracy
from surprise import Reader
from surprise.model_selection import train_test_split
from surprise.model_selection import cross_validate
from surprise.model_selection import GridSearchCV

import os
import numpy as np
import pandas as pd

########################
# simple movies dataset
########################

# path = '../'
# os.chdir(path)
trans = pd.read_csv('../Datasets/Toby/simplemovies-transactions.csv')
trans.columns = ['user','item','rating']

 # convert to surprise format
reader = Reader(rating_scale=(1,5)) # assumes datafile is: user, item, ratings (in this order)
data = Dataset.load_from_df(trans, reader)

################################################################
# build a model 
################################################################

trainset = data.build_full_trainset()  # use all data (ie no train/test split)

# select the model type, below are some examples, you can configure your own

algo = KNNBasic() # default method = User-based CF, default similarity is MSD (euclidean), default k =40
algo = KNNBasic(k=40,sim_options={'name': 'pearson'}) # User-based CF using pearson
algo = KNNBasic(sim_options={'name': 'cosine', 'user_based': False}) # item-based CF using cosine
algo = KNNWithMeans()  
algo = KNNWithMeans(k=40,sim_options={'name': 'pearson'}) 
algo = SVD(n_factors = 50) # simon funks algorithm, default is 100 factors)
algo = SVDpp(n_factors = 50) # an extension of SVD handling implicit ratings

algo.fit(trainset) # build the model

################################################################
# predict the rating for a specific user and item
################################################################

# select a target user
rawuid = 'Toby' 

# select an item (any one of the below)
rawiid = 'SnakesOnPlane' # was rated by Toby
rawiid = 'NightListener' # was not rated by Toby
rawiid = 'LadyinWater' # was not rated by Toby
rawiid = 'JustMyLuck' # was not rated by Toby

# convert user and items names (raw ids) into indexes (inner ids)
# (raw ids are the user & item names as given in the datafile, they can be ints or strings
# inner ids are indexes into the sorted rawids)
uid = trainset.to_inner_uid(rawuid); uid
iid = trainset.to_inner_iid(rawiid); iid

# if the actual rating is known it can be passed as an argument
realrating = dict(trainset.ur[uid])[iid]; realrating
pred = algo.predict(rawuid, rawiid, r_ui = realrating, verbose = True)

# if the actual rating is unknown use the below
pred = algo.predict(rawuid, rawiid); pred 

# FYI: can compare with prediction made using demolib (the library used in workshop1)
usersA, umap, imap = makeratingsmatrix(trans)
targetuser = usersA[umap[rawuid],]; targetuser
predictrating_UU(targetuser,usersA,imap[rawiid],simfun=pearsonsim)

# FYI: to help understand how predictions are made when using matrix factorisation we can 
# compute the prediction ourselves from the factorised matrices and the biases: pu,qi,bu,bi

# examine top-left part of the User and Item preference matrix
algo.pu[0:10,0:10] 
algo.qi[0:10,0:10]
# examine the learned biases (these are useful for cold-start)
algo.bu[0:10] # user mean ratings
algo.bi[0:10]# item mean ratings
algo.default_prediction() # the global mean rating

# examine the data for the target user & item
algo.pu[uid,] # user preferences
algo.qi[iid,]  # item preferences
algo.bu[uid] # user biases
algo.bi[iid] # item biases

# compute the prediction, this should agree with the above output from algo.predict()
pred = algo.bu[uid] + algo.bi[iid] + sum(algo.pu[uid,] * algo.qi[iid,]) + algo.default_prediction(); pred

################################################################
# to predict the ratings for all users on all unrated items
################################################################

 # create a testset by putting all of the unseen user/item events into a list
testset = trainset.build_anti_testset(); len(testset)
testset[1:20] # the rating shown is the global mean rating, the actual rating is unknown

# This makes predictions for all users - but may be slow for big datasets
predictions = algo.test(testset) ; len(predictions)
predictions

# to predict only the ratings for only the target user (specfied earlier by rawuid)
targetonly = list()
for uid, iid, r in testset:
    if (uid == rawuid):
        targetonly.append((uid, iid, r))
        
predictions = algo.test(targetonly) ; len(predictions)
predictions

################################################################
# to make recommendations for each user (the topN recommendations)
# we define a function called: get_top_n() which ranks the unseen items by their predicted rating 
# it takes as input the above rating predictions
# it returns a dictionary where keys are (raw) userids and 
# values are lists of tuples: [(raw item id, pred.rating),...] 
# see https://surprise.readthedocs.io/en/stable/FAQ.html
################################################################

from collections import defaultdict

def get_top_n(predictions, n=10):
    # First map the predictions to each user.
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))  
    # Then sort the predictions for each user and retrieve the k highest ones.
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True) # sort on predicted rating
        top_n[uid] = user_ratings[:n]
    return top_n

get_top_n(predictions)

###########################################################################
# To compute the accuracy of the rating predictions using a train/test split
###########################################################################

# usually we prefer the test set to be 10% to 30% of total data but for speed of this demo we keep it small
# if test_size parameter is float then it represents proportion of the data, if integer it represents absolute number 
trainset, testset  = train_test_split(data, test_size=0.002); len(testset)

# show stats about the split
testdf = pd.DataFrame(testset)
print('users,items in trainset=',trainset.n_users, trainset.n_items)
print('users,items in testset=',len(testdf.iloc[:,0].unique()),len(testdf.iloc[:,1].unique()))

# for an accurate test we must rebuild the model using the new training set
algo.fit(trainset)
preds = algo.test(testset)
accuracy.rmse(preds)
accuracy.mae(preds)

# run 5-fold cross-validation.
res = cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

# ########################################################################
# for comparison, we can repeat the same test using demolib 
# ########################################################################

usersA, umap, imap = makeratingsmatrix(trans)

 # blank out the test events in the training data
for (u,i,r) in testset:  usersA[umap[u],imap[i]] = np.nan

# # user-based CF (use whichever similarity measure gave you best results in workshop1)
simfun = pearsonsim
simfun = euclidsim
simfun = cosinesim
errs = computeErrs_UU(testset, usersA, umap, imap, simfun=simfun)
np.nanmean(abs(errs))

# item-based CF
simfun = euclidsim
itemsims = getitemsimsmatrix(usersA, simfun=simfun)
errs = computeErrs_II(testset, usersA, umap, imap, itemsims)
np.nanmean(abs(errs))


#########################
# movielens dataset
# FYI: this datset is also preloaded into the surprise package
# data = Dataset.load_builtin('ml-100k') 
########################

path = 'D:/datasets/movielens'
os.chdir(path)
trans = pd.read_csv('../Datasets/Movielens/u_data.csv') # movielens 100K file
trans = trans.iloc[:,0:3] # keep only first 3 columns
trans.columns = ['user','item','rating']

 # convert to surprise format
reader = Reader(rating_scale=(1,5)) 
data = Dataset.load_from_df(trans, reader)

# build a model
trainset = data.build_full_trainset()  # use all data (ie no train/test split)
algo = SVD(n_factors = 50) # simon funks algorithm, default is 100 factors)

algo.fit(trainset) # build the model

# pick a target user
rawuid = 7 

# get a list of all unseen items, then extract only the target
testset = trainset.build_anti_testset() 
targetonly = list()
for uid, iid, r in testset:
    if (uid == rawuid):
        targetonly.append((uid, iid, r))
        
# make predictions and recommendation for the target   
predictions = algo.test(targetonly)
get_top_n(predictions, n=10)

# compute MAE for a testset
trainset, testset  = train_test_split(data, test_size=0.002); len(testset)
algo.fit(trainset) # must rebuild the model
preds = algo.test(testset)
accuracy.mae(preds)

# how does this MAE compare with the MAE from workshop1?

#######################################
# book crossings dataset
# ratings are 1 to 10
# values of 0 are implicit (the book was read but not rated)
########################################

path = 'D:/datasets/Bookcrossings'
os.chdir(path)
trans = pd.read_csv('../Datasets/BookCrossings/BX-Book-Ratings.csv', sep=';', error_bad_lines=False, encoding="latin-1")
trans.columns = ['user','item','rating']

# remove implicit ratings
trans = trans[trans.rating != 0]
trans.shape

 # convert to surprise format
min(trans.rating)
max(trans.rating)
reader = Reader(rating_scale=(1,10)) 
data = Dataset.load_from_df(trans, reader)

# select a model
# note that user-based and item-based CF require more memory - dont work on fullBookcrossings dataset
algo = SVD(n_factors = 50)

# compute MAE for a testset
trainset, testset  = train_test_split(data, test_size=0.002); len(testset)
algo.fit(trainset)
preds = algo.test(testset)
accuracy.mae(preds)

# how does this MAE compare with the MAE from workshop1? (from the sampled dataset)

###################################################################

# IF YOU ARE AMBITIOUS AND HAVE TIME TRY COMPUTING PRECISION & RECALL FOR A GIVEN RATING THRESHOLD
# copy the code from:
# https://surprise.readthedocs.io/en/stable/FAQ.html


#############################################################
# some interesting reads...

# https://towardsdatascience.com/building-and-testing-recommender-systems-with-surprise-step-by-step-d4ba702ef80b

# to see SVD (funks alg) code go to...

# https://github.com/NicolasHug/Surprise/blob/master/surprise/prediction_algorithms/matrix_factorization.pyx

###############################################################

