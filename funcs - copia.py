import importlib
from os import defpath
from pprint import pprint
from re import X
import arbol
importlib.reload(arbol)
from arbol import *
import numpy as np
import random
import pandas as pd
MAX_DEPTH=3
MIN_DEPTH=2
depth=0
i=0
CONST_LIST=[SYMBOL,5*random.random ()]
def random_terminal():
	if random.random()<0.5:
		return GPTree(SYMBOL)
	else:
		return GPTree(5*random.random ())
def random_tree():
	return gen_tree(random.randint(MIN_DEPTH,MAX_DEPTH))
def gen_tree(depth=0):
	if depth<MIN_DEPTH:
		if depth==0:
			n1=GPTree(random.choice(FUNC_AR2_LIST))
		else:
			n1 = GPTree(random.choice(FUNC_LIST))
	elif depth>=MAX_DEPTH:
		n1= random_terminal()
	else:
		if random.random()<0.5:
			n1 = GPTree(random.choice(FUNC_LIST))
		else:
			n1= random_terminal()
	if n1.is_func():
		n1.left=gen_tree(depth+1)
		if n1.arity()==2:
			n1.right=gen_tree(depth+1)
	return n1		

def mutation_element(tree):
	node=tree.preorder[random.randint(0,len(tree.preorder)-1)]

	if node.is_func():
		if node.arity()==2:
				node.val=random.choice(FUNC_AR2_LIST)
		else:
				node.val=random.choice(FUNC_AR1_LIST)
	else:
		node.val=random.choice(CONST_LIST)

	return tree
def mutation_tree(tree):
	tree.pprint()

	node_parent=tree.preorder[random.randint(1,len(tree.preorder)-1)]

	node_parent.val=random.choice(FUNC_LIST)
	print(node_parent.arity())
	node_parent.left=gen_tree(random.randint(MIN_DEPTH+1,MAX_DEPTH))
	if node_parent.arity()==2:
		node_parent.right=gen_tree(random.randint(MIN_DEPTH+1,MAX_DEPTH))
	else:
		node_parent.right=None
	
	tree.pprint()
	return tree

def crossover(tree1,tree2):
	node1=tree1.preorder[random.randint(1,len(tree1.preorder)-1)]
	node2=tree2.preorder[random.randint(1,len(tree2.preorder)-1)]
	node1.val,node2.val=node2.val,node1.val
	node1.right,node2.right=node2.right,node1.right
	node1.left,node2.left=node2.left,node1.left
	return tree1,tree2

	
def evaluate(tree):
	tree.calculate_recursive(X)


def tournament(trees,fitness,K,elitism):
	for i in range(int(len(trees)*elitism)):
		l=len(trees)
		idx=random.sample(5,K)
		print(idx)
		values=np.argsort(fitness[idx])






	
csvfile = pd.read_csv('C:/Users/DMM/Downloads/unknown_function.csv')
x=csvfile['x'].values
x=x.reshape(-1,1)
print(x.shape)
y=csvfile['y'].values
y=y.reshape(-1,1)
print(y.shape)

a = gen_tree(depth)
b = gen_tree(depth)
tress=[a,b]
fitness=[1,5]
elitism=0.5
K=2
tournament(tress,fitness,K,elitism)
