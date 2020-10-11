# -*- coding: utf-8 -*-
"""
Library for demonstrating simple collaborative filtering
@author: barry shepherd
"""

import os
import math
import numpy as np
import pandas as pd
import time
from statistics import mean
from math import sqrt

# convert the transaction data (long data) into a ratings matrix (wide data)
# assume the first two columns are user and item names (which may be strings or integers and may not be contiguous)
# also generate two lookup tables to map user and item names into indexes for accessing the ratings matrix
def makeratingsmatrix(trans):
    trans = trans.iloc[:,0:3] # keep only first 3 columns
    trans.columns = ['user','item','rating']
    # create the mappings between user and item names (as in raw data) and the matrix row and column indexes
    unames  = np.sort(trans['user'].unique())
    inames  = np.sort(trans['item'].unique())
    umap = dict(zip(unames,[i for i in range(len(unames))]))
    imap = dict(zip(inames,[i for i in range(len(inames))]))
    # create the ratings matrix, use average if multiple raings exist for same (user,item)
    #users = trans.pivot(index='user', columns='item', values='rating').values # fast, but no averaging, rows & cols are alphnum order
    users = pd.pivot_table(trans, index=['user'], columns=['item'], values=['rating'],aggfunc=[mean]).values # slower
    return [users, umap, imap]

def head(arr,r=10,c=10):
    nr, nc = arr.shape
    with np.printoptions(threshold=np.inf):
        if type(arr) == np.ndarray:
            print(arr[0:min(r,nr),0:min(c,nc)])
        else:
            print(arr.iloc[0:min(r,nr),0:min(c,nc)])

def sparsity(arr):
    return float(np.isnan(arr).sum()*100)/np.prod(arr.shape)
    #return (1.0 - ( count_nonzero(arr) / float(arr.size) ))

def wtavg(vals, weights):
    xy = vals * weights
    weights = weights[np.isnan(xy) == False] 
    #if len(weights) == 0 : return np.nan
    if sum(weights) == 0 : return np.nan
    vals = vals[np.isnan(xy)==False]
    return sum(vals * weights)/sum(weights)
        
def pearsonsim(x,y):
    xy = x*y
    x = x[np.isnan(xy)==False]
    y = y[np.isnan(xy)==False]
    if(len(x)==0): return np.nan
    mx=mean(x)
    my=mean(y)
    rt = sqrt(sum((x-mx)**2)*sum((y-my)**2))
    if (rt == 0): return np.nan  #math.isnan(rt)==True or 
    return sum((x-mx)*(y-my))/rt
               
def cosinesim(x,y):
    xy = x*y
    x = x[np.isnan(xy)==False]
    y = y[np.isnan(xy)==False]
    if(len(x)==0): return np.nan
    rt = sqrt(sum(x**2)*sum(y**2))
    return sum(x*y)/rt

def euclidsim(x,y):
    xy = x*y
    x = x[np.isnan(xy)==False]
    y = y[np.isnan(xy)==False]
    z=(y-x)**2
    sz=sqrt(sum(z))
    return 1/(1+sz)

def euclidsimF(x,y):
    xy = x*y
    x = x[np.isnan(xy)==False]
    y = y[np.isnan(xy)==False]
    z=(y-x)**2
    return 1/(1+sum(z))

def getitemsimsmatrix(ratsmatrix,simfun):
    r,c = ratsmatrix.shape
    matrx = list([])
    for col1 in range(0,c):
        simrow = [0]*col1
        for col2 in range(col1,c):
            simrow.append(simfun(ratsmatrix[:,col1],ratsmatrix[:,col2]))
        matrx.append(simrow)
    matrx = np.array(matrx)
    matrx = matrx + matrx.T - np.diag(np.diag(matrx))
    return matrx
   
def predictrating_UU(targetrats, ratsmatrix, targetitemindx, simfun):
    return predictratings_UU(targetrats, ratsmatrix, doitems=[targetitemindx], simfun=simfun)[0]

def predictratings_UU(targetrats, ratsmatrix, doitems, simfun=pearsonsim):
    sims = list([])
    for row in ratsmatrix: sims.append(simfun(row,targetrats))
    sims = np.array(sims)
    with np.errstate(invalid='ignore'): sims[sims < 0] = np.nan
    rats = list([])
    for col in doitems: rats.append(wtavg(ratsmatrix[:,col],sims)) # assumes target rating is NA (if target in usersA)
    return np.array(rats)

def predictrating_II(targetrats, itemsims, targetitemid):
    return predictratings_II(targetrats, itemsims, doitems=[targetitemid])[0]

def predictratings_II(targetrats,itemsims,doitems):
    seenitems = np.isnan(targetrats)==False
    rats = list([])
    for row in doitems:
        rats.append(wtavg(targetrats[seenitems],itemsims[row,seenitems])) 
    return np.array(rats)

def getRecommendations_UU(targetrats, ratsmatrix, imap, simfun=pearsonsim,topN=5):
    itemnames=list(imap.keys())
    unseenitemids = np.where(np.isnan(targetrats)==True)[0]
    ratsA = predictratings_UU(targetrats, ratsmatrix, doitems=unseenitemids, simfun=simfun)
    rats = pd.DataFrame(ratsA,index=[itemnames[i] for i in unseenitemids],columns=['predrating'])
    rats = rats.sort_values(ascending = False, by=['predrating'])
    return rats[0:min(topN,len(rats))]
    
def getRecommendations_II(targetrats, itemsims, imap, topN=5):
    itemnames=list(imap.keys()) 
    unseenitemids = np.where(np.isnan(targetrats)==True)[0]
    ratsA = predictratings_II(targetrats,itemsims,doitems=unseenitemids)
    rats = pd.DataFrame(ratsA,index=[itemnames[i] for i in unseenitemids],columns=['predrating'])
    rats = rats.sort_values(ascending = False, by=['predrating'])
    return rats[0:min(topN,len(rats))]

# compute prediction errors (predicted rating - actual rating) for the test events (events ~ 'user,item,rating')
def computeErrs_UU(testevents, ratsmatrix, uids, iids, simfun=cosinesim):
    res = list([])
    for testevent in testevents:
        print('.', end = '')
        testuserindx = uids[testevent[0]]
        testitemindx = iids[testevent[1]]
        pred = predictrating_UU(ratsmatrix[testuserindx,],ratsmatrix,testitemindx,simfun=simfun)
        res.append(pred-testevent[2])
    return np.array(res)

def computeErrs_II(testevents, ratsmatrix, uids, iids, itemsims):
    res = list([])
    for testevent in testevents:
        print('.', end = '')
        testuserindx = uids[testevent[0]]
        testitemindx = iids[testevent[1]]
        pred = predictrating_II(ratsmatrix[testuserindx,],itemsims,testitemindx)
        res.append(pred-testevent[2])
    return np.array(res)

# returns the percentage ranking for each test event
# if itemsims is supplied then do item-based CF, else do user-based CF
def computePR(testevents, ratsmatrix, uids, iids, itemsims=False, simfun=cosinesim):
    res = list([])
    for testevent in testevents:
        print('.', end = '')
        testuserindx = uids[testevent[0]]
        if (type(itemsims) == bool):
            recs = getRecommendations_UU(ratsmatrix[testuserindx,], ratsmatrix, iids, simfun=simfun, topN=100000)
        else:
            recs = getRecommendations_II(ratsmatrix[testuserindx,], itemsims, iids, topN=100000)
        rkpc = ((recs.index.get_loc(testevent[1]) + 1)*100)/len(recs)
        res.append(rkpc)
    return np.array(res)

    

