'''
Matlab source code provided by Ruslan Salakhutdinov
More details can be found in his website
Python code modefied by zachary 2017-03-18
'''
import numpy as np
import random
import scipy.io as sio
import matplotlib.pyplot as plt 

def loadData(fileName):
	# load data from a Matlab data source
	# Triplets: {user_id, movie_id, rating}
	# train_vec: 900000x3 
	# probe_vec: 100209x3
	# number of users: 6040
	# number of movies: 3952
	mat_content = sio.loadmat("moviedata.mat")
	train_vec = mat_content['train_vec']
	probe_vec = mat_content['probe_vec']
	return train_vec, probe_vec

def pmf(train_vec, probe_vec, maxEpoch = 60, num_feat = 10, epsilon = 50,
										 lambdaReg = 0.01, momentum = 0.8):
	# length of training data
	pairs_tr = len(train_vec)
	# length of validation data
	pairs_pr = len(probe_vec)
	# number of users
	num_p = 6040
	# number of movies
	num_m = 3952
	# number of barches
	numBatches = 9
	# number training per batch
	N = 100000
	# user feature vectors
	w_P = 0.1 * np.random.randn(num_p, num_feat)
	# movie feature vectors
	w_M = 0.1 * np.random.randn(num_m, num_feat)
	w_P_inc = np.zeros((num_p, num_feat))
	w_M_inc = np.zeros((num_m, num_feat))

	mean_rating = np.mean(train_vec[:, 2])

	# to store training error history
	err_valid = []
	err_train = []

	for epoch in xrange(maxEpoch):
		rr = np.random.permutation(pairs_tr)
		train_vec = train_vec[rr, :]
		for batch in xrange(1, numBatches + 1):
			print("epoch %d batch %d" %(epoch, batch))
			# user_id and movie_id begin with 1
			aa_p = train_vec[(batch - 1)*N : batch * N, 0] - 1
			aa_m = train_vec[(batch - 1)*N : batch * N, 1] - 1
			rating = train_vec[(batch - 1)*N : batch * N, 2]
			# default prediction is the mean rating
			rating = rating - mean_rating
			# compute predictions
			pred_out = np.sum(w_M[aa_m, :] * w_P[aa_p, :], axis = 1)
			f = np.sum((pred_out - rating)**2 + 
				0.5 * lambdaReg *(np.sum(w_M[aa_m,:]**2 + w_P[aa_p,:]**2, axis = 1)), axis = 0)
			# compute gradient
			temp = pred_out - rating
			temp = temp[:, np.newaxis]
			IO = np.tile(2 * temp, (1, num_feat))
			Ix_m = IO * w_P[aa_p, :] + lambdaReg * w_M[aa_m,:]
			Ix_p = IO * w_M[aa_m, :] + lambdaReg * w_P[aa_p,:]

			dw_M = np.zeros((num_m, num_feat))
			dw_P = np.zeros((num_p, num_feat))

			for i in range(N):
				dw_M[aa_m[i], :] = dw_M[aa_m[i], :] + Ix_m[i, :]
				dw_P[aa_p[i], :] = dw_P[aa_p[i], :] + Ix_p[i, :]
			# update movie and user features
			w_M_inc = momentum * w_M_inc + 1.0 * epsilon * dw_M/N
			w_M = w_M - w_M_inc
			w_P_inc = momentum * w_P_inc + 1.0 * epsilon * dw_P/N
			w_P = w_P - w_P_inc

		# compute prediction after parameter updates
		pred_out = np.sum(w_M[aa_m, :] * w_P[aa_p, :], axis = 1)
		f_s = np.sum((pred_out - rating)**2 + 0.5
		 * lambdaReg *(np.sum(w_M[aa_m,:]**2 + w_P[aa_p,:]**2, axis = 1)), axis = 0)
		err_train.append(np.sqrt(f_s/N))

		# compute prediction on validation set
		NN = pairs_pr

		aa_p = probe_vec[:, 0] - 1
		aa_m = probe_vec[:, 1] - 1
		rating = probe_vec[: ,2]

		pred_out = np.sum(w_M[aa_m, :] * w_P[aa_p, :], axis = 1) + mean_rating
		ff = pred_out > 5
		pred_out[ff] = 5
		ff = pred_out < 1
		pred_out[ff] = 1
		err_valid.append(np.sqrt(np.sum((pred_out - rating)**2, axis = 0)/NN))

		print("---epoch %d batch %d Training RMSE %f Test RMSE %f" %(epoch, batch,
			err_train[epoch], err_valid[epoch]))
	plotPicture(err_train, err_valid, maxEpoch)

def plotPicture(err_train, err_valid, maxEpoch):
	plt.figure()
	plt.xlabel("epoch")
	plt.ylabel("RMSE")
	plt.grid(True)
	plt.title("Error curve")
	plt.plot(range(maxEpoch), err_train, '-*r', linewidth = 2)
	plt.plot(range(maxEpoch), err_valid, '-sk', linewidth = 2)
	plt.legend(["Train", "Test"])
	plt.savefig("plot.png")
	plt.show()

if __name__ == '__main__':
	fileName = "moviedata.mat"
	train_vec, probe_vec = loadData(fileName)
	pmf(train_vec, probe_vec)

