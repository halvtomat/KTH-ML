#!/usr/bin/env python3
import math
import numpy as np
from scipy.optimize import minimize
from scipy import stats
import pandas as pd

class Svm:
	N = 0
	C = 0
	K = None
	t = []
	x = []
	P = []
	alpha = []
	b = 0

	def __init__(self, N, C, t, x):
		self.N = N
		self.C = C
		self.K = self.lin_kernel
		self.t = t
		self.x = x
	
	def pre_compute(self):
		self.P = np.zeros((self.N, self.N))
		for i in range(self.N):
			for j in range(i, self.N):
				value = self.t[i] * self.t[j] * self.K(self.x[i], self.x[j])
				self.P[i][j] = value
				self.P[j][i] = value

	def lin_kernel(self, a, b):
		return np.dot(a, b)

	def poly_kernel(self, a, b):
		p = 3
		return math.pow(np.dot(a, b) + 1, p)
	
	def radial_kernel(self, a, b):
		s = 3
		return math.exp(-math.pow(math.dist(a, b), 2)/(2 * math.pow(s, 2))) 

	def zerofun(self, a):
		return np.dot(a, self.t)

	def objective(self, a):
		return np.sum([ np.dot(a[i] * a, self.P[i]) for i in range(self.N) ]) / 2 - np.sum(a)

	def calc_b(self):
		return np.sum([ self.alpha[i] * self.t[i] * self.K(self.x[0], self.x[i]) for i in range(len(self.alpha))]) - self.t[0]

	def indicator(self, s):
		return np.sum([ self.alpha[i] * self.t[i] * self.K(s, self.x[i]) for i in range(len(self.alpha))]) - self.b

	def remove_zero_values(self):
		indices = np.where(self.alpha < 0.00001)
		self.alpha = np.delete(self.alpha, indices)
		self.x = np.delete(self.x, indices, 0)
		self.t = np.delete(self.t, indices)

	def build_model(self):
		self.pre_compute()
		print("Pre compute done")
		B = [(0, self.C) for _ in range(self.N)]
		XC = {'type':'eq', 'fun':self.zerofun}
		start = np.zeros(self.N)
		ret = minimize(self.objective, start, bounds=B, constraints=XC)
		print("Minimize done")
		self.alpha = ret['x']
		self.remove_zero_values()
		self.b = self.calc_b()
	
class NB(object):
	def __init__(self):
		self.trained = False

	def computePrior(self, labels, W=None):
		Npts = labels.shape[0]
		if W is None:
			W = np.ones((Npts,1))/Npts
		else:
			assert(W.shape[0] == Npts)
		classes = np.unique(labels)
		prior = np.zeros((np.size(classes), 1))

		for i,c in enumerate(classes):
			prior[i] = [np.sum(W[labels == c,:]) / np.sum(W)]

		return prior

	def mlParams(self, X, labels, W=None):
		assert(X.shape[0]==labels.shape[0])
		Npts,Ndims = np.shape(X)
		classes = np.unique(labels)
		Nclasses = np.size(classes)

		if W is None:
			W = np.ones((Npts,1))/float(Npts)

		mu = np.zeros((Nclasses,Ndims))
		sigma = np.zeros((Nclasses,Ndims,Ndims))

		for i,c in enumerate(classes):
			x = X[labels == c,:]
			w = W[labels == c,:]
			mu[i] = np.sum(x * w, axis=0) / np.sum(w)
			sigma[i] = np.diag(np.sum(w * (x - mu[i]) ** 2, axis=0) / np.sum(w))

		return mu, sigma

	def classifyBayes(self, X, prior, mu, sigma):
		Npts = X.shape[0]
		Nclasses,Ndims = np.shape(mu)
		logProb = np.zeros((Nclasses, Npts))

		for i in range(Nclasses):
			a = -0.5 * np.log(np.linalg.det(sigma[i]))
			c = np.log(prior[i])
			for j in range(Npts):
				b = -0.5 * np.linalg.multi_dot([(X[j,:] - mu[i]), np.linalg.inv(sigma[i]), (X[j,:] - mu[i]).T])
				logProb[i][j] = a + b + c		

		h = np.argmax(logProb,axis=0)
		return h

	def trainClassifier(self, X, labels, W=None):
		self.prior = self.computePrior(labels, W)
		self.mu, self.sigma = self.mlParams(X, labels, W)
		self.trained = True

	def classify(self, X):
		return self.classifyBayes(X, self.prior, self.mu, self.sigma)

class BoostClassifier:
	def __init__(self, base_classifier, T=10):
		self.base_classifier = base_classifier
		self.T = T
		self.trained = False

	def trainBoost(self, base_classifier, X, labels, T=10):
		Npts,Ndims = np.shape(X)

		classifiers = []
		alphas = [] 

		wCur = np.ones((Npts,1))/float(Npts)

		for i in range(0, T):
			bc = base_classifier()
			bc.trainClassifier(X, labels, wCur)
			classifiers.append(bc)
			vote = classifiers[-1].classify(X)
			error = np.sum(wCur[vote != labels]) + 0.0001
			alphas.append(0.5 * (np.log(1 - error) - np.log(error)))
			wCur[vote == labels] *= np.exp(-alphas[-1])
			wCur[vote != labels] *= np.exp(alphas[-1])
			wCur /= np.sum(wCur)
			print(i, "/", T-1)
		return classifiers, alphas

	def classifyBoost(self, X, classifiers, alphas, Nclasses):
		Npts = X.shape[0]
		Ncomps = len(classifiers)

		if Ncomps == 1:
			return classifiers[0].classify(X)
		else:
			votes = np.zeros((Npts,Nclasses))

			for i,c in enumerate(classifiers):
				v = c.classify(X)
				for j,p in enumerate(v):
					votes[j][p] += alphas[i]
				print(i, "/", Ncomps-1)
			return np.argmax(votes,axis=1)

	def trainClassifier(self, X, labels):
		self.nbr_classes = np.size(np.unique(labels))
		self.classifiers, self.alphas = self.trainBoost(self.base_classifier, X, labels, self.T)
		self.trained = True

	def classify(self, X):
		return self.classifyBoost(X, self.classifiers, self.alphas, self.nbr_classes)

def read_input(path):
	return pd.read_csv(path)

def write_output(data):
	pd.DataFrame(data).to_csv('result.csv', index=False)

def format_input(data):
	new = data.drop(data.columns[0], axis=1)
	new['et'] = new['et'].replace(['A', 'B', 'I', 'W'], [0, 1, 2, 3])
	return new

def format_training(data, targets):
	#indexes_to_keep = (np.abs(stats.zscore(data)) < 3).all(axis=1) Removes outliers if used
	return data.to_numpy(), targets_to_numbers(targets.to_numpy())

def format_testing(data, targets):
	return data.to_numpy(), targets_to_numbers(targets.to_numpy())

def partition(data, fraction=0.7):
	training = data.sample(frac=fraction)
	testing = data.drop(training.index)
	return training, testing

def extract_targets(data):
	return data.drop('y', axis=1), data['y']

def targets_to_numbers(targets):
	return np.array([1 if t == 'L' else 0 for t in targets])

def numbers_to_targets(targets):
	return np.array(['L' if t == 1 else 'C' for t in targets])

def test(classifier, data, split=0.8):
	training, testing = partition(data, split)
	print("Partitioning done")
	x_train, t_train = extract_targets(training)
	x_train, t_train = format_training(x_train, t_train)
	cl = classifier
	cl.trainClassifier(x_train, t_train)
	print("Training completed")
	x_test, t_test = extract_targets(testing)
	t_test2 = t_test.to_numpy()
	x_test, t_test = format_testing(x_test, t_test)
	res = cl.classify(x_test)
	print("targets =", t_test2)
	print("results =", numbers_to_targets(res))
	print("res==test =", res==t_test)
	return np.sum(res==t_test)/len(t_test)

def run_classification(classifier, training, evaluating):
	x, t = extract_targets(training)
	x, t = format_training(x, t)
	cl = classifier
	cl.trainClassifier(x, t)
	print("Training completed")
	write_output(numbers_to_targets(cl.classify(evaluating)))
	print("Classifying completed")

def main():
	data = read_input('TrainOnMe-4.csv')
	print("Input read\nLength of data = ", len(data))
	data = format_input(data)

	#res = test(NB(), data)
	#print("Testing results", res)

	eval = read_input('EvaluateOnMe-4.csv')
	print("Eval read\nLength of eval = ", len(eval))
	eval = format_input(eval).to_numpy()
	run_classification(NB(), data, eval)


if __name__ == '__main__':
	main()