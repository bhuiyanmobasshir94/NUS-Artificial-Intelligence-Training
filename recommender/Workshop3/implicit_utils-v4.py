# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 14:32:56 2020
@author: issbas
"""

import pandas as pd
import numpy as np
import scipy.sparse as sparse
import implicit
import os

###########################
#The deskdrop dataset
###########################

# Deskdrop allows employees to share relevant articles with their peers, and collaborate around them.
# The data contains about 73k users interactions on more than 3k public articles shared in the platform, 
# see https://towardsdatascience.com/building-a-collaborative-filtering-recommender-system-with-clickstream-data-dffc86c8c65

path = 'D:/datasets/deskdrop'
path = 'D:/datasets/kaggle/deskdrop'
os.chdir(path)

# interaction events for individual users, eventype ~ view, like, bookmark, follow, comment
interactions_df = pd.read_csv('users_interactions.csv')
interactions_df.drop(['userAgent', 'userRegion', 'userCountry'], axis=1, inplace=True)
interactions_df.head(5)

# load article info so we can obtain the article titles
articles_df = pd.read_csv('shared_articles.csv')
articles_df.drop(['authorUserAgent', 'authorRegion', 'authorCountry'], axis=1, inplace=True)
articles_df = articles_df[articles_df['eventType'] == 'CONTENT SHARED']
articles_df.drop('eventType', axis=1, inplace=True)
articles_df.head(5)

# join on contentId to obtain the article titles
trans = pd.merge(interactions_df[['contentId','personId', 'eventType']], articles_df[['contentId', 'title']], how = 'inner', on = 'contentId')

# Create an implict rating called eventStrength based on the type of the interaction with the article
# E.g, assume a bookmark indicates a higher interest than a like etc.
# To do this, create a dictionary to associate each eventType with a weight.
trans['eventType'].value_counts()
 
event_type_strength = {
   'VIEW': 1.0,
   'LIKE': 2.0, 
   'BOOKMARK': 3.0, 
   'FOLLOW': 4.0,
   'COMMENT CREATED': 5.0,  
}
trans['eventStrength'] = trans['eventType'].apply(lambda x: event_type_strength[x])

# if a user has multiple interactions on the same content then sum the strengths
# Group eventStrength together with person and content.
trans = trans.drop_duplicates()
trans.columns = ['item','user','eventType','title','rating']
trans = trans.groupby(['user', 'item', 'title']).sum().reset_index()
trans.sample(5)

# map to contiguous int ranges (note that the raw user and items ids are very very long integers , often negative)
trans,umap,imap = maptrans(trans)
trans.head(10) 

#Create two matrices, one for fitting the model (content-person) and one for recommendations (person-content)
#Create using sparse.csr_matrix((data,(row,column)))
sparse_item_user = sparse.csr_matrix((trans['rating'].astype(float), (trans['item'],trans['user'])))
sparse_user_item = sparse.csr_matrix((trans['rating'].astype(float), (trans['user'],trans['item'])))

#Initialize the Alternating Least Squares (ALS) recommendation model.
model = implicit.als.AlternatingLeastSquares(factors=20, regularization=0.1, iterations=50)

# Set matrix to double for the ALS function to run properly.
# note that each time the model is fitted may result in slightly different results (diff factor matrices)
alpha = 15
data = (sparse_item_user * alpha).astype('double')
model.fit(data)

###############################################
# Use the trained item properties to find the top 10 most similar articles for content_id = 450, 
# this article title=“Google’s fair use victory is good for open source”, it talks about Google and open source.
#################################################

item_id = 450
trans.title[trans.item == item_id]

 # use the implicit library built-in
similar = model.similar_items(item_id)
for item, score in similar: print(score,'\t',trans.title.loc[trans.item == item].iloc[0])

# FYI - we can do the calc ourselves (should get identical results)
# we use the item-properties matrix (Q) to compute nearest neighbours using cosine similarity
user_vecs = model.user_factors; user_vecs.shape  # user preferences (the P matrix)
item_vecs = model.item_factors; item_vecs.shape  # item properties (the Q matrix)

similar = findsimilaritems(item_id, item_vecs)
for item, score in similar: print(score,'\t',trans.title.loc[trans.item == item].iloc[0])
    
###################################################
# Make recommendations for specific users
###################################################

user_id = 50

 # use the implicit library built-in
recommendations = model.recommend(user_id, sparse_user_item, filter_already_liked_items=True)
for item, score in recommendations: 
    print(f'{score:0.5f}','\t',trans.title.loc[trans.item == item].iloc[0])
    
# use own function (do the matrix calculations ourselves, should get identical results)
recommendations = recommend(user_id, sparse_user_item, user_vecs, item_vecs)
for item, score in recommendations: print(f'{score:0.5f}','\t',trans.title[trans.item == item].iloc[0])
 
# Do these recommendations make sense? Examine the top 10 articles this person has rated.
trans[trans.user == user_id].sort_values(by=['rating'], ascending=False)[['rating', 'title']].head(10)

# try another person
user_id = 1
recommendations = recommend(user_id, sparse_user_item, user_vecs, item_vecs)
for item, score in recommendations: print(f'{score:0.5f}','\t',trans.title[trans.item == item].iloc[0])
trans[trans.user == user_id].sort_values(by=['rating'], ascending=False)[['rating', 'title']].head(10)
