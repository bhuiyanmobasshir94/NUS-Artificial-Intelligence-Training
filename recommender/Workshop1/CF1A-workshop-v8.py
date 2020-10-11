# -*- coding: utf-8 -*-
"""
Demonstrate simple collaborative filtering
@author: barry shepherd
"""

#########################################################
# simple movies dataset
#########################################################

# load the data
path = '../Datasets/Toby'
os.chdir(path)
trans = pd.read_csv('simplemovies-transactions.csv')
ratmatrix, uids, iids = makeratingsmatrix(trans)
ratmatrix.shape
ratmatrix

targetname = "Toby" 
targetrats = ratmatrix[uids[targetname],] # note: index into ratingsmatrix is by index, e.g. user "10" ~ index 9
targetrats

# UU recommendations for Toby with pearson shd be: 3.35 (night), 2.83 (lady), 2.53 (luck)
getRecommendations_UU(targetrats, ratmatrix, iids, simfun=pearsonsim, topN = 10)

# compute item-item similarity matrix (with timer)
tic = time.perf_counter()
itemsims = getitemsimsmatrix(ratmatrix, simfun=euclidsimF) # use euclidsimF to agree with book/slide calcs
print(f"time {time.perf_counter() - tic:0.4f} seconds")

head(itemsims)

# II recommendations for Toby with euclideanF shd be: 3.183 (night), 2.598 (luck), 2.473 (lady)
getRecommendations_II(targetrats, itemsims, iids, topN = 10)

# an aside illustrating python built-in correlation functions
np.corrcoef(ratmatrix,rowvar=False) # corr. between movies (columns), numpy corrcoef() does not handle na's
pd.DataFrame(ratmatrix).corr() # corr. between movies (columns), pandas can handle na's
print(getitemsimsmatrix(ratmatrix, simfun=pearsonsim))  # for comparison
pd.DataFrame(ratmatrix).T.corr() # corr. between users

# lets try pre-normalising the data
rowmeans = np.nanmean(ratmatrix,axis=1); rowmeans
normratmatrix = ratmatrix.copy()
for i in range(ratmatrix.shape[0]):  # iterate over rows
    normratmatrix[i] = normratmatrix[i] - rowmeans[i]
head(normratmatrix)

# redo the UU recommendations
targetrats = normratmatrix[uids[targetname],] 
recs = getRecommendations_UU(targetrats, normratmatrix, iids, simfun=pearsonsim, topN = 10); recs # the normalised rating predictions
recs + rowmeans[uids[targetname]] # the unnormalised rating predictions


#########################################################
# TESTING the recommendations 
#########################################################

# load a bigger data set to demo a split into training and test sets
trans = pd.read_csv('simplemovies-transactions-moreusers.csv'); trans
ratmatrix, uids, iids = makeratingsmatrix(trans)
ratmatrix
ratmatrix.shape

# extract a testset from the rating events
testsize = 10
testevents = trans.sample(n=testsize).values.tolist(); testevents

# blank out the testset ratings in the rating matrix
# (this is an easy alternative to blanking them out at the time of the prediction)
for (uname,iname,rating) in testevents: ratmatrix[uids[uname],iids[iname]] = np.nan 

# try using each of these in turn 
simfun = pearsonsim 
simfun = cosinesim
simfun = euclidsim

errs = computeErrs_UU(testevents, ratmatrix, uids, iids, simfun=simfun); errs
np.nanmean(abs(errs))

# calc the item similarity matrix
# try using each of these in turn
simfun = euclidsimF 
simfun = euclidsim
simfun = cosinesim

tic = time.perf_counter()
itemsims = getitemsimsmatrix(ratmatrix, simfun = simfun)
print(f"time {time.perf_counter() - tic:0.4f} seconds")

errs = computeErrs_II(testevents, ratmatrix, uids, iids, itemsims)
np.nanmean(abs(errs))

#########################################################
# movielens dataset
#########################################################

path = '../Datasets/Movielens'
os.chdir(path)
trans = pd.read_csv('u_data.csv') # movielens 100K file (user and itemids start at 1)
trans.drop('datetime',axis=1,inplace=True)
ratmatrix, uids, iids = makeratingsmatrix(trans)
ratmatrix.shape
sparsity(ratmatrix) # show % that is empty
ratmatrix

# select any user at random to make recommendations to, e.g.:
targetname = 10 # a movielens user
targetrats = ratmatrix[uids[targetname],] 
uurecs = getRecommendations_UU(targetrats, ratmatrix, iids, simfun=pearsonsim, topN = 20); uurecs
itemsims = getitemsimsmatrix(ratmatrix, simfun=euclidsimF) # takes ~ 20-30secs
iirecs = getRecommendations_II(targetrats, itemsims, iids, topN = 20) ; iirecs

# An interesting aside:
# For a given target user, how many of the topN recommendations from user-based CF are 
# also in the topN from item-based CF? (try topN = 20 to start with).
# To do this convert the recommended items into sets and compute the intersection, e.g.
uuset = set(uurecs.index)
iiset = set(iirecs.index)
uuset.intersection(iiset)

# Proceed as above to create train/test sets and to compute MAE using user-based & item-based CF
# and to explore performance of the various similarity measures.
# Note1: for increased test accuracy use a bigger test size, e.g. 100 
# (even bigger is better but increases the testing time)
# Note2: for item-based CF, you must compute the item similarity matrix AFTER the testevents have been blanked in the ratings matrix

testsize = 100

#########################################################
# Jester dataset
#########################################################

# dataset2
path = "../Datasets/Jester"
os.chdir(path)
trans = pd.read_csv("jester_ratings.dat", sep='\s+',header=0)
ratmatrix, uids, iids = makeratingsmatrix(trans)
ratmatrix.shape
sparsity(ratmatrix) # show % that is empty
head(ratmatrix)

# Proceed as above to create train/test sets and to compute MAE using user-based & item-based CF
# and to explore performance of the various similarity measures.
# Note: pearsonsim may be too slow to test mfor this dataset
# What MAE is reasonable given that the ratings range -10->10 is much larger than movielens ?

#########################################################
# book crossings dataset
#########################################################

path = "../Datasets/BookCrossings"
os.chdir(path)
trans = pd.read_csv("BX-Book-Ratings.csv", sep=';', error_bad_lines=False, encoding="latin-1")
trans.columns = ['user','item','rating']
trans.shape

# remove implicit ratings
trans = trans[trans.rating != 0]
trans.shape

# this fails since expanded ratings matrix is too big to fit
ratmatrix, uids, iids = makeratingsmatrix(trans) 

# reduce dataset size
min_item_ratings = 10 # book popularity threshold 
popular_items = trans['item'].value_counts() >= min_item_ratings
popular_items = popular_items[popular_items].index.tolist(); len(popular_items)  # get list of popular items

min_user_ratings = 10 # user activity threshold
active_users = trans['user'].value_counts() >= min_user_ratings
active_users = active_users[active_users].index.tolist(); len(active_users) # get list of active users

print('original data: ',trans.shape)
trans = trans[(trans['item'].isin(popular_items)) & (trans['user'].isin(active_users))] # apply the filter
print('new data: ', trans.shape)

# converting to a matrix now succeeds
ratmatrix, uids, iids = makeratingsmatrix(trans)
ratmatrix.shape

# Proceed as above to create train/test sets 
# and to compute MAE using user-based & item-based CF
# and to explore performance of the various similarity measures.
# What MAE is reasonable given that the ratings range is 1 to 10?
# Note: computing the item-similarity matrix may take some time (~ 10mins) if the 
# user activity and book popularity thresholds are set at 10. 
# If you are impatient then increasing the thesholds to 20 reduces the dataset size and speeds up computation time



