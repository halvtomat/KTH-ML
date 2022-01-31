#!/usr/bin/env python3

import csv
import random
import numpy as np

import files.dectrees.python.monkdata as m
import files.dectrees.python.dtree as dt
import files.dectrees.python.drawtree_qt5 as dr

def partition(data, fraction):
	ldata = list(data)
	random.shuffle(ldata)
	breakPoint = int(len(ldata) * fraction)
	return ldata[:breakPoint], ldata[breakPoint:]

def prune_tree(tree, validation):
	pruned_trees = dt.allPruned(tree)
	score = dt.check(tree, validation)
	pruned_scores = [dt.check(t, validation) for t in pruned_trees]
	if(max(pruned_scores) > score):
		best_tree = pruned_trees[pruned_scores.index(max(pruned_scores))]
		return prune_tree(best_tree, validation)
	else:
		return tree

def test_fraction(training, test, fraction):
	train, valid = partition(training, fraction)
	tree = dt.buildTree(train, m.attributes)
	pruned = prune_tree(tree, valid)

	return dt.check(pruned, test)

def main():
	fractions = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
	training_sets = [m.monk1, m.monk3]
	testing_sets = [m.monk1test, m.monk3test]
	with open('data.csv', 'w') as file:
		writer = csv.writer(file)
		for i in [0,1]:
			row = []
			for f in fractions:
				row.append(f)
			writer.writerow(row)
			data = []
			print("\n\n")
			print("FRAC\tMEAN\tVAR")
			for f in fractions:
				frac_data = [test_fraction(training_sets[i], testing_sets[i], f) for _ in range(100)]
				mean = np.mean(frac_data)
				var = np.var(frac_data)
				print(f, round(mean, 4), round(var, 4), sep='\t')
				data.append(frac_data)
			for j in range(100):
				row = []
				for k in range(len(fractions)):
					row.append(data[k][j])
				writer.writerow(row)
			unpruned_tree = dt.buildTree(training_sets[i], m.attributes)
			print("UNPRUN", round(dt.check(unpruned_tree, testing_sets[i]), 4), sep='\t')

if __name__ == "__main__":
	main()