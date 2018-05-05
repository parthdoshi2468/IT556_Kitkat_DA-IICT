import sys
import numpy as np
import pandas as pd
import math
from math import sqrt
from numpy import linalg as LA
import csv
import random

from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import mean_squared_error

names = ['user_id', 'item_id', 'rating', 'timestamp']
df =pd.read_csv('ml-100k/u.data', sep='\t', names=names)
df.head()

names1 = ['item_id', 'item_name', 'date', 'kuchhbhi','url', 'g1','g2','g3','g4','g5','g6','g7','g8','g9','g10','g11','g12','g13','g14','g15','g16','g17','g18','g19']
item_d =pd.read_csv('ml-100k/u.item', sep='|', names=names1)
item_d.head()

item_data=np.array(item_d)

n_users = df.user_id.unique().shape[0]
n_items = df.item_id.unique().shape[0]
print str(n_users) + ' users'
print str(n_items) + ' items'

ratings = np.zeros((n_users, n_items))
for row in df.itertuples():
	ratings[row[1]-1, row[2]-1] = row[3]

def train_test_split(ratings):
    test = np.zeros(ratings.shape)
    train = ratings.copy()
    c=0
    for user in xrange(ratings.shape[0]):
        test_ratings = np.random.choice(ratings[0, :].nonzero()[0], 
                                        size=250, 
                                        replace=False)
        train[user, test_ratings] = 0.
        test[user, test_ratings] = ratings[user, test_ratings]
        
    # Test and training are truly disjoint
    assert(np.all((train * test) == 0)) 
    return train, test

train, test= train_test_split(ratings)
print "Testing Data" ,np.count_nonzero(test)
print "Training Data" , np.count_nonzero(train)

user_genre = np.empty((n_users, 19))
for i in range(0,n_users):
	for j in range(0,19):
		user_genre[i,j]=5

def reward(x,y,z=0.12):
	return 0.15*y*math.exp(-1*(x-5)*(x-5)/y)

def penalty(x,y):
	return 0.15*(-1*y+6)*math.exp(-1*(x-5)*(x-5)/(6-y))

def nor(x,y):
	return 0.8*y*math.exp(-1*x*x/5)

for u_id in range(0,n_users):
	for i_id in range(0,n_items):
		given_rating=train[u_id,i_id]
		if (given_rating>0):
			for i in range(0,19):
				if (item_data[i_id,i+5]==1):
					if(given_rating>3):
						change=reward(user_genre[u_id,i],given_rating)
						#print(user_genre[u_id,i],given_rating,temp)
						user_genre[u_id,i]=user_genre[u_id,i]+change
					elif(given_rating<3):
						change=penalty(user_genre[u_id,i],given_rating)
						#print(user_genre[u_id,i],given_rating,temp)
						user_genre[u_id,i]=user_genre[u_id,i]-change

def predict(ratings, similarity, type='user'):
    if type == 'user':
        mean_user_rating = ratings.mean(axis=1)
        # We use np.newaxis so that mean_user_rating has same format as ratings
        ratings_diff = (ratings - mean_user_rating[:, np.newaxis])
        pred = mean_user_rating[:, np.newaxis] + similarity.dot(ratings_diff) / np.array(
            [np.abs(similarity).sum(axis=1)]).T
    elif type == 'item':
        pred = ratings.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])
    return pred

def get_mse(pred, actual):
    # Ignore nonzero terms.
    pred = pred[actual.nonzero()].flatten()
    actual = actual[actual.nonzero()].flatten()
    return mean_squared_error(pred, actual)


def rmse(actual, prediction):  # Root mean squared error calculation
    return sqrt(mean_squared_error(actual, prediction))

user_similarity = pairwise_distances(ratings, metric='cosine')

user_prediction = predict(ratings, user_similarity, type='user')

print 'User-based CF MSE: ' + str(rmse(user_prediction, test))



for u_id in range(0,n_users):
	for i_id in range(0,n_items):
		if(test[u_id,i_id]!=0):
			predicted_rating=user_prediction[u_id,i_id]
			genre_count=0
			genre_like=0

			for i in range(0,19):
				if(item_data[i_id,i+5]==1):
					genre_count=genre_count+1
					genre_like=genre_like+user_genre[u_id,i]
			genre_like=genre_like/genre_count
			
			if(genre_like>=5.0):
				user_prediction[u_id,i_id]=predicted_rating+nor(predicted_rating,genre_like-5)
				if (user_prediction[u_id,i_id]>=5 and ratings[u_id,i_id]==0):
					c11+=1

			elif(genre_like<=5.0):
				user_prediction[u_id,i_id]=predicted_rating-nor(predicted_rating-5,5-genre_like)
			

			if(user_prediction[u_id,i_id]<0):
				user_prediction[u_id,i_id]=0
			if(user_prediction[u_id,i_id]>5):
				user_prediction[u_id,i_id]=5.0
	#print u_id,c11
print 'Hybrid CF MSE: ' + str(rmse(user_prediction, test))

for u_id in range(0,n_users):
	c11=0
	for i_id in range(0,n_items):
		if(test[u_id,i_id]==0):
			predicted_rating=user_prediction[u_id,i_id]
			genre_count=0
			genre_like=0

			for i in range(0,19):
				if(item_data[i_id,i+5]==1):
					genre_count=genre_count+1
					genre_like=genre_like+user_genre[u_id,i]
			genre_like=genre_like/genre_count
			
			if(genre_like>=6.0):
				user_prediction[u_id,i_id]=predicted_rating+nor(predicted_rating,genre_like-5)

			elif(genre_like<=4.0):
				user_prediction[u_id,i_id]=predicted_rating-nor(predicted_rating-5,5-genre_like)
			

			if(user_prediction[u_id,i_id]<0):
				user_prediction[u_id,i_id]=0
			if(user_prediction[u_id,i_id]>5):
				user_prediction[u_id,i_id]=5.0


def global_ratings(ratings):
    glo_rat=np.zeros(n_items)
    for i_id in range(0,n_items):
        count=0
        sum=0
        for u_id in range(0,n_users):
            if(train[u_id][i_id]==0):
                continue
            elif(train[u_id][i_id]!=0):
                sum=sum+train[u_id][i_id]
                count = count+1
            val=sum/count
        if(count>10):
            glo_rat[i_id]=val
        elif(count<=5):
            glo_rat[i_id]=0.0
    return glo_rat 

glo_rat=global_ratings(train)


tot_items=item_d.item_id.unique().shape[0]

seen_movies = np.zeros((n_users,tot_items))

def find_seen():
	names = ['user_id', 'item_id', 'rating', 'timestamp']
	df =pd.read_csv('ml-100k/u.data', sep='\t', names=names)
	df.head()

	for row in df.itertuples():
		seen_movies[row[1]-1,row[2]-1] = 1



top_k = np.zeros(10)

def get_top_movies(user_prediction,user):
	#while 1:
	a1 = np.zeros(tot_items)
	a2 = np.zeros(tot_items)

	for j in range(1,tot_items+1):
		a1[j-1] = j-1
		a2[j-1] = user_prediction[user-1][j-1]

	r = zip(a1,a2)
	s=r[:]
	
	r.sort(key=lambda x: x[1])
	k=0

	for i in range(1,tot_items+1):

		#if(int(seen_movies[user-1][int(r[tot_items-i][0])])==0):
		if(ratings[user-1,int(r[tot_items-i][0])]==0):
			top_k[k]=r[tot_items-i][0]
			k = k+1

			if(k==6):
				break

	id1,id2 = global_recommended(user-1)
	top_k[k]=id1
	top_k[k+1]=id2
	print 'global',
	print id1+1,
	print id2+1
	id3,id4 = serendipity(user-1)
	top_k[k+2]=id3
	top_k[k+3]=id4
	print 'Serendipity',
	print id3+1,
	print id4+1
	names1 = ['item_id', 'item_name', 'date', 'kuchhbhi','url', 'g1','g2','g3','g4','g5','g6','g7','g8','g9','g10','g11','g12','g13','g14','g15','g16','g17','g18','g19']
	item_d =pd.read_csv('ml-100k/u.item', sep='|', names=names1)
	item_d.head()	

	top_k.sort()
	cc=0


	for row in item_d.itertuples():
		if(row.item_id==top_k[cc]+1):
			print cc,
			print row.item_id,
			print row.item_name
			cc=cc+1

			if(cc==10):
				break;

	#re_id=input("Select One")
	#re_rating=input("Give Rating")
	#seen_movies[user-1,top_k[re_id]]=1
	#if(re_id==id1 or re_id==id2):
			
def global_recommended(u_id):
    #count=0
    u_g_pre=np.zeros(19)
    #for u_id in range(0,n_users):
    for i in range(0,19):
        if(user_genre[u_id][i]>4.0):
            u_g_pre[i]=1
        else:
        	u_g_pre[i]=0

    max1_rat=0
    max2_rat=0
    id1=-1
    id2=-1
    g_alpha=0.9
    for u_g in range(0,19):
        for i_g in range(0,n_items):
            if(u_g_pre[u_g]==1 and item_data[i_g][u_g+5]==1):
            	mul=(g_alpha*glo_rat[i_g])+((1-g_alpha)*(user_genre[u_id][u_g]/2))
                if(mul>=max1_rat):
                	#print glo_rat[i_g],user_genre[u_id][u_g],mul,i_g
                	max2_rat=max1_rat
                	id2=id1
                	max1_rat=mul
                	id1=i_g
                elif(mul>max2_rat):
                	#print glo_rat[i_g],user_genre[u_id][u_g],mul,i_g
                	max2_rat=mul
                	id2=i_g
    return id1,id2                
    
def serendipity(u_id):
	count=0;
	while(count!=2):
		id1 = random.randint(1,1680)
		if(train[u_id][id1]==0):
			count = count + 1
		id2 = random.randint(1,1680)
		if(train[u_id][id2]==0 and id2!=id1):
			count=count+1
	return id1,id2


find_seen()
userid=input("Enter id of the user to whom you want to recommend.")
get_top_movies(user_prediction,userid)

#for i in range(0,19):
#	print i,user_genre[userid-1,i]

#print 'Hybrid RMSE: ' + str(mean_absolute_error(user_prediction, test)