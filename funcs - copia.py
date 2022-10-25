from os import defpath
from pprint import pprint
from re import X
from arbol import *
import numpy as np
import random
import pandas as pd
import warnings
from copy import deepcopy
# esto para quitar warnings de np.arrays de Nodes
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 

def mae(y, y_pred, w=None):
    return np.average(np.abs(y-y_pred), weights=w)

def mse(y, y_pred, w=None):
    return np.average(np.power(y-y_pred,2), weights=w)

def rmse(y, y_pred, w=None):
    return np.sqrt(mse(y, y_pred, w=w))

PROB_FUNCTION=0.5
PROB_SYMBOL=0.5

PROB_CROSS=0.02
PROB_MUT_TREE=0.02
PROB_MUT_ELEMENT=0.02

MAX_DEPTH=3
MIN_DEPTH=2
depth=0

ERROR_FUNC = rmse
# para penalizar tamaños grandes de arbol en fitness
FIT_ADJUST_SIZE = 0.02

def random_terminal():
	if random.random()<PROB_SYMBOL:
		return SYMBOL
	else:
		return 5*random.random()
		
def random_tree():
	return gen_tree(random.randint(MIN_DEPTH,MAX_DEPTH))

def gen_tree(depth=0):
	if depth<MIN_DEPTH:
		if depth==0:
			n1 = GPTree(random.choice(FUNC_AR2_LIST))
		else:
			n1 = GPTree(random.choice(FUNC_LIST))
	elif depth>=MAX_DEPTH:
		n1 = GPTree(random_terminal())
	else:
		if random.random()<PROB_FUNCTION:
			n1 = GPTree(random.choice(FUNC_LIST))
		else:
			n1 = GPTree(random_terminal())
	if n1.is_func():
		n1.left=gen_tree(depth+1)
		if n1.arity()==2:
			n1.right=gen_tree(depth+1)
	return n1		

def mutation_element(tree):
	tree = deepcopy(tree)
	node=np.random.choice(tree.preorder[0:])
	if node.is_func():
		if node.arity()==2:
				node.val=random.choice(FUNC_AR2_LIST)
		else:
				node.val=random.choice(FUNC_AR1_LIST)
	else:
		node.val=random_terminal()
	return tree

def mutation_tree(tree):
	tree = deepcopy(tree)
	node_parent=np.random.choice(tree.preorder[1:])
	node_parent.val=random.choice(FUNC_LIST)

	node_parent.left=gen_tree(random.randint(MIN_DEPTH+1,MAX_DEPTH))
	if node_parent.arity()==2:
		node_parent.right=gen_tree(random.randint(MIN_DEPTH+1,MAX_DEPTH))
	else:
		node_parent.right=None
	return tree

def crossover(tree1,tree2):
	tree1, tree2 = deepcopy(tree1), deepcopy(tree2)

	node1=np.random.choice(tree1.preorder[1:])
	node2=np.random.choice(tree2.preorder[1:])
	
	node1.val,node2.val=node2.val,node1.val
	node1.right,node2.right=node2.right,node1.right
	node1.left,node2.left=node2.left,node1.left
	return tree1,tree2
	
def eval_fitness(tree, x, y, w=None):
    y_pred = tree.calculate_recursive(x)
    return ERROR_FUNC(y,y_pred,w=w)*(1+FIT_ADJUST_SIZE*tree.size)

eval_fitness_vec=np.vectorize(eval_fitness, excluded=('x','y','w'))

def fill_K_best(P, fitness,Pe):
	return P[np.argpartition(fitness, Pe)[:Pe]]

def tournament(P, fitness, K, N):
	l=len(fitness)
	winners = []
	for i in range(N):		
		idx = np.random.randint(0,l,size=K)
		winner=P[np.argmin(fitness[idx])]
		winners.append(winner)
	return np.fromiter(winners, dtype=GPTree)


def target_func(x): # para probar si funciona, funcion facil
    return x*x*x*x + x*x*x + x*x + x + 1

def generate_dataset(): # generate 101 data points from target_func
    x_list, y_list = [], []
    for x in range(-100,101,2): 
        x /= 100
        x_list.append(x)
        y_list.append(target_func(x))
    return np.array(x_list), np.array(y_list)

	
#csvfile = pd.read_csv('C:/Users/DMM/Downloads/unknown_function.csv')
#x=csvfile['x'].values
#x=x.reshape(-1,1)
#print(x.shape)
#y=csvfile['y'].values
#y=y.reshape(-1,1)
#print(y.shape)
x, y = generate_dataset()

M = 100
tourn = 0.03
elitism = 0.2
Pe = int(elitism*M)
K = int(tourn*M)

P = np.fromiter([gen_tree() for i in range(M)], dtype=GPTree)
fitness = eval_fitness_vec(P, x=x, y=y)
for j in range(50):
	new_P = []
	P_elite = fill_K_best(P, fitness, Pe)
	new_P.extend(P_elite)
	P_tournament = tournament(P, fitness, K, M-Pe)
	i = 0
	while len(new_P) < M:
		if random.random() < PROB_CROSS and i<(M-Pe)-1:
			new_P.extend(crossover(P_tournament[i], P_tournament[i+1]))
			i += 2
		elif random.random() < PROB_MUT_TREE:
			new_P.append(mutation_tree(P_tournament[i]))
			i += 1
		elif random.random() < PROB_MUT_ELEMENT:
			new_P.append(mutation_element(P_tournament[i]))
			i += 1
		else:
			new_P.append(P_tournament[i])
			i += 1
	P = np.fromiter(new_P, dtype=GPTree)
	fitness = eval_fitness_vec(P, x=x, y=y)
best_f_ind = np.argmin(fitness)

print(P[best_f_ind])
print(fitness[best_f_ind])



