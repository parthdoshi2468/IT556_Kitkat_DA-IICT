import os
import surprise
from guppy import hpy
from surprise import BaselineOnly
from surprise import Dataset
from surprise import Reader
from surprise import SVD
from surprise.model_selection import cross_validate
# path to dataset file
if __name__ == '__main__':
	file_path = os.path.expanduser('C:/Users/admin/Desktop/Data/music/ratings_digital_music(full).csv')
	reader = Reader(line_format='user item rating timestamp', sep=',')

	data = Dataset.load_from_file(file_path, reader=reader)
	algo = SVD()
	cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
	h = hpy()
	print h.heap()