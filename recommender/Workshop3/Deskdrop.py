# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import scipy.sparse as sparse
import implicit
import os

def maptrans(trans):
    uniqueusers = np.sort(trans['user'].unique())
    uniqueitems = np.sort(trans['item'].unique())
    umap = dict(zip(uniqueusers,[i for i in range(len(uniqueusers))])) 
    imap = dict(zip(uniqueitems,[i for i in range(len(uniqueitems))]))
    trans['user'] = trans.apply(lambda row: umap[row['user']], axis = 1) 
    trans['item'] = trans.apply(lambda row: imap[row['item']], axis = 1) 
    return (trans,umap,imap)

# Code
path = '../Datasets/Deskdrop'
os.chdir(path)

interactions_df = pd.read_csv('users_interactions.csv')
interactions_df.drop(['userAgent', 'userRegion', 'userCountry'], axis=1, inplace=True)

articles_df = pd.read_csv('shared_articles.csv')
articles_df.drop(['authorUserAgent', 'authorRegion', 'authorCountry'], axis=1, inplace=True)
articles_df = articles_df[articles_df['eventType'] == 'CONTENT SHARED']
articles_df.drop('eventType', axis=1, inplace=True)

trans = pd.merge(interactions_df[['contentId','personId', 'eventType']], articles_df[['contentId', 'title']], how = 'inner', on = 'contentId')

event_type_strength = {
   'VIEW': 1.0,
   'LIKE': 2.0, 
   'BOOKMARK': 3.0, 
   'FOLLOW': 4.0,
   'COMMENT CREATED': 5.0,  
}
trans['eventStrength'] = trans['eventType'].apply(lambda x: event_type_strength[x])

trans = trans.drop_duplicates()
trans.columns = ['item','user','eventType','title','rating']
trans = trans.groupby(['user', 'item', 'title']).sum().reset_index()
trans.sample(5)

trans,umap,imap = maptrans(trans)

sparse_item_user = sparse.csr_matrix((trans['rating'].astype(float), (trans['item'],trans['user'])))
sparse_user_item = sparse.csr_matrix((trans['rating'].astype(float), (trans['user'],trans['item'])))

model = implicit.als.AlternatingLeastSquares(factors=20, regularization=0.1, iterations=50)

alpha = 15
data = (sparse_item_user * alpha).astype('double')
model.fit(data)

item_id = 450
similar = model.similar_items(item_id)
for item, score in similar: 
    print(score,'\t',trans.title.loc[trans.item == item].iloc[0], "\n")
print()

item_id = 291
similar = model.similar_items(item_id)
for item, score in similar: 
    print(score,'\t',trans.title.loc[trans.item == item].iloc[0], "\n")
print()

user_id = 1
recommendations = model.recommend(user_id, sparse_user_item, filter_already_liked_items=True)
for item, score in recommendations: 
    print(f'{score:0.5f}','\t',trans.title[trans.item == item].iloc[0], "\n")
print()

user_id = 50
recommendations = model.recommend(user_id, sparse_user_item, filter_already_liked_items=True)
for item, score in recommendations: 
    print(f'{score:0.5f}','\t',trans.title[trans.item == item].iloc[0], "\n")