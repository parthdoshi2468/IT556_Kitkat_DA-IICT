import numpy as np
import pandas as pd
import csv
import sklearn.preprocessing as sk
from gensim.corpora import MmCorpus
from gensim.test.utils import get_tmpfile
from numpy.linalg import matrix_rank
import gensim
import gensim.models.lsimodel as ls
from sklearn.decomposition import TruncatedSVD
from sklearn.random_projection import sparse_random_matrix
from sklearn.utils.extmath import randomized_svd


ratings_list = [i.strip().split(",") for i in open('rating.csv', 'r').readlines()]



ratings_df = pd.DataFrame(ratings_list, columns = ['UserID', 'SongID', 'Rating'], dtype = float)


ratings_df.loc[:,'Rating'] = sk.minmax_scale( ratings_df.loc[:,'Rating'] )


R_df = ratings_df.pivot(index = 'UserID', columns ='SongID', values = 'Rating').fillna(0)
U_R_matrix = R_df.as_matrix()

Z=gensim.matutils.Dense2Corpus(U_R_matrix, documents_columns=True)

lsi=ls.LsiModel(Z, num_topics=3)

print("U")
print(lsi.projection.u)

print("S")
print(lsi.projection.s)
Sigma_mat = np.diag(lsi.projection.s)


print("VT")
V = gensim.matutils.corpus2dense(lsi[Z], len(lsi.projection.s)).T / lsi.projection.s
print(V)

a=np.matmul(lsi.projection.u,Sigma_mat)
approx=np.matmul(a,V.T)


print("Approximation error")
print((approx - U_R_matrix).sum())