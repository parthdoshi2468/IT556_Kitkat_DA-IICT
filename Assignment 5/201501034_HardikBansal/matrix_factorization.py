import os
import surprise
import numpy as np
from guppy import hpy
from surprise import BaselineOnly
from surprise import Dataset
from surprise import Reader
from surprise import SVD
from surprise.model_selection import cross_validate

class MatrixFacto(surprise.AlgoBase):
    skip_train = 0
    '''A basic rating prediction algorithm based on matrix factorization.'''
    
    def __init__(self, learning_rate, n_epochs, n_factors):
        
        self.lr = learning_rate  # learning rate for SGD
        self.n_epochs = n_epochs  # number of iterations of SGD
        self.n_factors = n_factors  # number of factors
        
    def train(self, trainset):

        
        print('Fitting data with SGD...')
        
        p = np.random.normal(0, .1, (trainset.n_users, self.n_factors))
        q = np.random.normal(0, .1, (trainset.n_items, self.n_factors))
        
        # SGD procedure
        for _ in range(self.n_epochs):
            for u, i, r_ui in trainset.all_ratings():
                err = r_ui - np.dot(p[u], q[i])
                # Update vectors p_u and q_i
                p[u] += self.lr * err * q[i]
                q[i] += self.lr * err * p[u]

        
        self.p, self.q = p, q
        self.trainset = trainset

    def estimate(self, u, i):

        if self.trainset.knows_user(u) and self.trainset.knows_item(i):
            return np.dot(self.p[u], self.q[i])
        else:
            return self.trainset.global_mean
if __name__ == '__main__':

    file_path = os.path.expanduser('C:/Users/admin/Desktop/Data/music/ratings_digital_music(21633).csv')
    reader = Reader(line_format='user item rating timestamp', sep=',')

    data = Dataset.load_from_file(file_path, reader=reader)
    data.split(5)  # split data for 2-folds cross validation

    algo = MatrixFacto(learning_rate=.01, n_epochs=10, n_factors=10)
    surprise.evaluate(algo, data, measures=['RMSE'])
    h = hpy()
    print h.heap()

