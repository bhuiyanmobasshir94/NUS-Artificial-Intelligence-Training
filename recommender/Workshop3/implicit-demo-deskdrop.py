# -*- coding: utf-8 -*-
"""
@author: issbas
"""

import pandas as pd
import numpy as np
import implicit
import scipy.sparse as sparse
import os
import random  

pd.set_option('display.max_columns',10)

os.environ['MKL_NUM_THREADS'] = '1'   # to prevent a warning message

# map the user and item names to contiguous integers and also return the maps
def maptrans(trans):
    uniqueusers = np.sort(trans['user'].unique())
    uniqueitems = np.sort(trans['item'].unique())
    umap = dict(zip(uniqueusers,[i for i in range(len(uniqueusers))])) # this maps username -> index
    imap = dict(zip(uniqueitems,[i for i in range(len(uniqueitems))])) # this maps itemname -> index
    trans['user'] = trans.apply(lambda row: umap[row['user']], axis = 1) 
    trans['item'] = trans.apply(lambda row: imap[row['item']], axis = 1) 
    return (trans,umap,imap)

#return list of similar items, use the item-properties matrix (Q) to do nearest neighbour using cosine similarity
def findsimilaritems(item, item_vecs, n_similar=10):
    #Calculate the item vector norms (the vector lengths)
    item_norms = np.sqrt((item_vecs * item_vecs).sum(axis=1))
    #Calculate the (cosine) similarity score: do dot product of selected content with every other content
    #Note: cosine sim = A.B/(norm(A)*norm(B)), since B (item 450) is the same for every item A, we can ignore its norm in this calc
    simscores = item_vecs.dot(item_vecs[item]) / item_norms
    #Get the top 10 contents (do a sort)
    top_idx = np.argpartition(simscores, -n_similar)[-n_similar:]
    #Create a descending list of content-score tuples of most similar articles with this article.
    similar = sorted(zip(top_idx, simscores[top_idx]/item_norms[item]), key=lambda x: -x[1])
    return (similar)

#return the top 10 recommendations chosen based on the person / content vectors 
#for contents never interacted with for any given person.
def recommend(user, sparse_user_item, userprefs, itemprops, num_items=10):

    # create a template vector, where unrated items = 1, rated items =0
    existing_ratings = sparse_user_item[user,:].toarray() # Get existing ratings for target
    existing_ratings = existing_ratings.reshape(-1) + 1  # Add 1 to everything, so items with no rating = 1
    existing_ratings[existing_ratings > 1] = 0  # make items already rated = 0

    # Get dot product of the target user preferences and all item properties ~ P[user]*transpose(Q)
    predrats = userprefs[user,:].dot(itemprops.T)
    
    # Items already rated have their predictions multiplied by zero (ie eliminated)
    predrats = predrats * existing_ratings 

    # Sort into descending order of predicted rating and select the topN item indexes
    itemids = np.argsort(predrats)[::-1][:num_items]
    
    # Start empty list to store items and scores
    recs = []
    for item in itemids: recs.append((item, predrats[item]))
    return recs


def implicit_testusers(testset, userprefs, itemprops, debug=False):
    errs = list([])
    #tic = time.perf_counter()
    for (indx,(uname,iname,rating)) in testset.iterrows():
        if (debug): print('.', end = '')
        err = abs(userprefs[uname,:].dot(itemprops[iname,:]) - rating)
        errs.append(err)
    #print(f"\ntime {time.perf_counter() - tic:0.4f} seconds")  
    return(errs)
    
    
#def ahead(arr,r=7,c=7):
#    with np.printoptions(threshold=np.inf):
#        print(arr[0:r,0:c])

#def sparsity(arr):
#    return np.isnan(arr).sum()/np.prod(arr.shape)
#    #1.0 - ( count_nonzero(arr) / float(arr.size) )
    
##################################################
# ASIDE
# sparse matrix investigations
    """
sparse_user_item.toarray()
sparse_item_user.toarray()

# nan is stored as an explicit value, else its assumes specified values are zeros
X_train = np.array([[ 1., -1.,  np.nan],[ 2.,  0.,  0.],[ 0.,  1., -1.]])
x = sparse.csr_matrix(X_train)
x.toarray()
x.nnz
x[0,2]

rows = np.array([0,1,2,3])
cols = np.array([0,1,2,3])
vals = np.array([0,2,np.nan,4])
x = sparse.csr_matrix((vals.astype(float),(rows,cols)))
x.toarray()
x.nnz
x.nonzero
x[0,0] * x[0,1] 
  """  
 