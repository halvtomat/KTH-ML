#!/usr/bin/env python3
import random, math, sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

t = []
K = None
N = 0
C = 0
b = 0
x = []
P = []
alpha = []

def pre_compute():
	P = np.zeros((N, N))
	for i in range(N):
		for j in range(i,N):
			value = t[i] * t[j] * K(x[i], x[j])
			P[i][j] = value 
			P[j][i] = value 

def lin_kernel(a, b):
	return np.dot(a, b)

def zerofun(a):
	return np.dot(a, t)

def objective(a):
	return np.sum([ np.dot(a[i]*a, P[i]) for i in range(N) ]) / 2 - np.sum(a)

def calc_b():
	return np.sum([ alpha[i] * t[i] * K(x[0], x[i]) for i in range(len(alpha))]) - t[0]

def indicator(s):
	return np.sum([ alpha[i] * t[i] * K(s, x[i]) for i in range(len(alpha))]) - b

def main():
	N = sys.argv[1] if len(sys.argv) > 1 else 100
	C = sys.argv[2] if len(sys.argv) > 2 else None
	B = [(0, C) for _ in range(N)]
	K = lin_kernel
	XC = {'type':'eq', 'fun':zerofun}
	start = np.zeros(N)

	ret = minimize(objective, start, bounds=B, constraints=XC)

	alpha = ret['x']

if __name__ == '__main__':
	main()