#!/usr/bin/env python3
import files.dectrees.python.dtree as dt
import files.dectrees.python.monkdata as m

def main(): 
	ent1 = dt.entropy(m.monk1)
	ent2 = dt.entropy(m.monk2)
	ent3 = dt.entropy(m.monk3)

	print("MONK1: ", ent1)
	print("MONK2: ", ent2)
	print("MONK3: ", ent3)

if __name__ == "__main__":
	main()