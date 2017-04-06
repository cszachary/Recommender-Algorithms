# The implement of BPR(Bayesian Personalized Ranking) in python

import numpy as np
import matplotlib.pyplot as plt
from preprocess import *

class BPR:
	'''
	Bayesian Personalized Ranking Matrix Factorization(BPR-MF)
	'''
	def __init__(self, n_factors = 100, n_iterations = 30, learning_rate = 0.01,
	 			 lambda_user = 0.05, lambda_item = 0.05):
		self.n_factors = n_factors
		self.n_iterations = n_iterations
		self.learning_rate = learning_rate
		self.lambda_user = lambda_user
		self.lambda_item = lambda_item

	def init(self, data):
		self.U = np.random.rand(self.n_users, self.n_factors)
		self.I = np.random.rand(self.n_items, self.n_factors)
		self.bI = np.zeros(self.n_items)

	def update(self, u, i, j):
		uF = np.copy(self.U[u,:])
		iF = np.copy(self.I[i,:])
		jF = np.copy(self.I[j,:])
		s = self.sigmoid(np.dot(uF, iF) - np.dot(uF, jF) + self.bI[i] - self.bI[j])
		c = 1.0 -s
		self.U[u,:] += self.learning_rate * (c * (iF - jF) - self.lambda_user * uF)
		self.I[i,:] += self.learning_rate * (c * uF - self.lambda_item * iF)
		self.I[j,:] += self.learning_rate * (c * (-uF) - self.lambda_item * jF)
		loss = self.lambda_user * np.dot(self.U[u,:], self.U[u,:]) 
		loss = loss + self.lambda_item * np.dot(self.I[i,:], self.I[i,:])
		loss = loss + self.lambda_item * np.dot(self.I[j,:], self.I[j,:])
		loss = 0.5 * loss - np.log(s)
		return loss

	def fit(self, data):
		'''
		Train the model
		'''
		self.n_users = data.shape[0]
		self.n_items = data.shape[1]

		self.init(data)
		his_error = []
		for it in range(self.n_iterations):
			c = []
			for _ in xrange(data.nnz):
				u = np.random.randint(0, self.n_users - 1)
				i = np.random.choice(data[u].indices)
				j = self.samplingNegativeItem(data[u].indices)
				err = self.update(u, i, j)
				c.append(err)
			print 'iteration {0}: loss = {1}'.format(it, np.mean(c))
			his_error.append(np.mean(c))
		plt.figure()
		plt.xlabel('iteration')
		plt.ylabel('loss')
		plt.grid(True)
		plt.title('BPR')
		plt.plot(range(self.n_iterations), his_error, '-sk', linewidth = 2)
		plt.savefig('plot.png')
		plt.show()

	def samplingNegativeItem(self, user_items):
		j = np.random.randint(0, self.n_items -1)
		while j in user_items:
			j = np.random.randint(0, self.n_items -1)
		return j

	def predict(self, u, i):
		return self.bI[i] + np.dot(self.U[u,:], self.I[i,:])

	def sigmoid(self, x):
		return 1.0/(1.0 + np.exp(-x))

if __name__ == '__main__':
	fileName = "ratings.dat"
	data = genTrainData(fileName)
	model = BPR()
	model.fit(data)
