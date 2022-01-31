#!/usr/bin/env python3
import files.dectrees.python.monkdata as m
import files.dectrees.python.dtree as dt

def main():
	tree1 = dt.buildTree(m.monk1, m.attributes)
	tree2 = dt.buildTree(m.monk2, m.attributes)
	tree3 = dt.buildTree(m.monk3, m.attributes)

	print("MONK1\t", dt.check(tree1, m.monk1), "\t", dt.check(tree1, m.monk1test), sep='')
	print("MONK2\t", dt.check(tree2, m.monk2), "\t", dt.check(tree2, m.monk2test), sep='')
	print("MONK3\t", dt.check(tree3, m.monk3), "\t", dt.check(tree3, m.monk3test), sep='')

if __name__ == "__main__":
	main()