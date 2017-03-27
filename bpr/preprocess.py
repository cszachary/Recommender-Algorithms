# preprocess data
import numpy as np
from scipy import sparse

def genTrainData(fileName):
	mat = np.zeros((6040, 3952), dtype = int)
	f = open(fileName, "r")
	lines = f.readlines()
	f.close()
	for line in lines:
		data = line.split("::")
		uid = int(data[0]) - 1
		mid = int(data[1]) - 1
		mat[uid, mid] = 1

	m = sparse.csr_matrix(mat)
	return m