
import pandas as pd
import numpy as np

import sklearn.preprocessing as sk
from sklearn.decomposition import TruncatedSVD
from sklearn.random_projection import sparse_random_matrix
from sklearn.utils.extmath import randomized_svd

ratings_df = pd.read_csv('rating.csv', dtype={'rating': float})
print (ratings_df)

print (ratings_df.head())

ratings_df.loc[:,'rating'] = sk.minmax_scale(ratings_df.loc[:,'rating'] )

print(ratings_df.loc[:,'rating'])
print (ratings_df)
print (ratings_df.head())


R_df = ratings_df.pivot(index = 'user_id', columns ='song_id', values = 'rating').fillna(0)
print(R_df.head())

R = R_df.as_matrix()


U, S, VT = randomized_svd(R,n_components=2)

svd = TruncatedSVD(n_components=20, n_iter=7)
svd.fit(R)
print("U")
print(U)
print("S")
print(S)
print("VT")
print(VT)


print(svd.explained_variance_ratio_)  

print(svd.components_)  

print(svd.singular_values_) 