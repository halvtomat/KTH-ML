#!/usr/bin/env python3
import files.dectrees.python.monkdata as m
import files.dectrees.python.dtree as dt

def main():
	gain1 = []
	gain2 = []
	gain3 = []
	for a in m.attributes:
		gain1.append(dt.averageGain(m.monk1, a))
		gain2.append(dt.averageGain(m.monk2, a))
		gain3.append(dt.averageGain(m.monk3, a))

	print("Gain\tA1\tA2\tA3\tA4\tA5\tA6")
	print("MONK1", end='')
	for a in gain1:
		print("\t", round(a, 4), sep='', end='')
	print("\nMONK2", end='')
	for a in gain2:
		print("\t", round(a, 4), sep='', end='')	
	print("\nMONK3", end='')
	for a in gain3:
		print("\t", round(a, 4), sep='', end='')
	print('')	

if __name__ == "__main__":
	main()