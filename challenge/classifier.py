#!/usr/bin/env python3
import random, math, sys
import numpy as np
from scipy.optimize import minimize
import csv
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
		B = [(0, self.C) for _ in range(self.N)]
		XC = {'type':'eq', 'fun':self.zerofun}
		start = np.zeros(self.N)
		ret = minimize(self.objective, start, bounds=B, constraints=XC)
		self.alpha = ret['x']
		self.remove_zero_values()
		self.b = self.calc_b()
	
	def test(self, data, targets):
		correct = 0
		total = 0
		for i,x in enumerate(data):
			if self.indicator(x) == targets[i]:
				correct += 1
			total += 1
		return correct/total

def read_input():
	cols = list(pd.read_csv('TrainOnMe-4.csv', nrows=1))
	cols.remove('et')
	return pd.read_csv('TrainOnMe-4.csv', usecols=cols).to_numpy()

def partition(data, fraction=0.01):
	np.random.shuffle(data)
	breakPoint = int(len(data) * fraction)
	return data[:breakPoint], data[breakPoint:]

def extract_targets(data):
	return data[:,:-1], data[:,-1]

def targets_to_numbers(targets):
	return np.array([1 if t == 'L' else -1 for t in targets])

def numbers_to_targets(targets):
	return np.array(['L' if t == 1 else 'C' for t in targets])

def main():
	C = float(sys.argv[1]) if len(sys.argv) > 1 else None

	data = read_input()
	print("Input read\nLength of data = ", len(data))
	training, testing = partition(data)
	print("Partitioning done")
	x, t = extract_targets(training)
	print("Targets extracted")
	t = targets_to_numbers(t)
	N = len(t)

	svm = Svm(N, C, t, x)
	svm.build_model()
	print("Model built")

	x_test, t_test = extract_targets(testing)
	t_test = targets_to_numbers(t_test)

	res = svm.test(x_test, t_test)
	print("Testing results", res)

if __name__ == '__main__':
	main()