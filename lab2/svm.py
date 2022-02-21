#!/usr/bin/env python3
import random, math, sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

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

def gen_test_data(N):
	width = 0.2
	classA = np.concatenate(
		(np.random.randn(int(N/4), 2) * width + [1.5, 1.5],
		 #np.random.randn(int(N/8), 2) * width + [-2.0, -2.0],
		 #np.random.randn(int(N/8), 2) * width + [2.0, -2.0],
		 np.random.randn(int(N/4), 2) * width + [-1.5, 1.5]))
	classB = np.random.randn(int(N/2), 2) * width + [0.0, 0.0]
	inputs = np.concatenate((classA, classB))
	targets = np.concatenate(
		(np.ones(classA.shape[0]),
		-np.ones(classB.shape[0])))
	permute = list(range(inputs.shape[0]))
	random.shuffle(permute)
	inputs = inputs[permute, :]
	targets = targets[permute]
	return targets, inputs, classA, classB

def plot_data(a, b, xgrid, ygrid, grid):
	plt.plot([p[0] for p in a],
			 [p[1] for p in a],
			 'b.')
	plt.plot([p[0] for p in b],
			 [p[1] for p in b],
			 'r.')
	plt.axis('equal')
	plt.contour(xgrid, ygrid, grid,
				(-1.0, 0.0, 1.0),
				colors=('red', 'black', 'blue'),
				linewidths=(1, 3, 1))
	plt.show()


def main():
	N = int(sys.argv[1]) if len(sys.argv) > 1 else 100
	C = float(sys.argv[2]) if len(sys.argv) > 2 else None
	np.random.seed(100)

	t, x, a, b = gen_test_data(N)

	svm = Svm(N, C, t, x)
	svm.build_model()

	xgrid = np.linspace(-5, 5)
	ygrid = np.linspace(-4, 4)
	grid = np.array([[svm.indicator([x, y])
									for x in xgrid]
									for y in ygrid])
	
	plot_data(a, b, xgrid, ygrid, grid)

if __name__ == '__main__':
	main()