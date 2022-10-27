from funcs import FUNC_LIST, FUNC_AR1_LIST, FUNC_AR2_LIST
from gptree import *
from tqdm import tqdm 
import numpy as np
import random

def mae(y, y_pred, w=None):
    return np.average(np.abs(y-y_pred), weights=w)

def mse(y, y_pred, w=None):
    return np.average(np.power(y-y_pred,2), weights=w)

def rmse(y, y_pred, w=None):
    return np.sqrt(mse(y, y_pred, w=w))


class GP():
    def __init__(self, 
                M=500, 
                tourn=0.03, 
                elitism=0.1, 
                const_range=(-1,1), 
                min_depth=1, 
                max_depth=10,
                func_list=FUNC_LIST,
                prob_func=None,
                prob_node_symb=0.5,
                prob_node_func=0.5,  
                probs_tree_op=[0.9,0.01,0.01], 
                error_fun=rmse, 
                depth_penalty=0, 
                calc_weights=False):

        self.M = M
        self.tourn, self.K_tourn = tourn, min([1,int(M*tourn)])
        self.elitism, self.K_elite = elitism, int(M*elitism)        
        self.const_range = const_range
        self.min_depth, self.max_depth = min_depth, max_depth
        self.func_list = func_list
        self.prob_func = prob_func if prob_func is not None else [1/len(func_list)]*len(func_list)
        self.prob_node_symb, self.prob_node_func = prob_node_symb, prob_node_func        
        self.probs_tree_op = probs_tree_op
        if len(probs_tree_op)<4: self.probs_tree_op.append(1-sum(probs_tree_op))
        self.error_fun, self.depth_penalty = error_fun, depth_penalty
        self.calc_weights = calc_weights

        ar1_mask, ar2_mask = np.isin(self.func_list,FUNC_AR1_LIST), np.isin(self.func_list,FUNC_AR2_LIST)
        self.func_ar1_list, self.func_ar2_list = np.asarray(self.func_list)[ar1_mask], np.asarray(self.func_list)[ar2_mask]
        self.prob_func_ar1, self.prob_func_ar2 = np.asarray(self.prob_func)[ar1_mask], np.asarray(self.prob_func)[ar2_mask]
        self.prob_func_ar1, self.prob_func_ar2 = self.prob_func_ar1/sum(self.prob_func_ar1), self.prob_func_ar2/sum(self.prob_func_ar2)

        self.P, self.fitness = None, None
        self.total_generations = 0
        self.best_fitness, self.mean_fitness = [], []
        self.best_trees = []
        self.P_max_depth, self.P_mean_depth = [], []

    def random_terminal(self):
        if random.random()<self.prob_node_symb:
            return SYMBOL
        else:
            return np.random.uniform(self.const_range[0],self.const_range[1])
    
    def random_func(self, arity=None):
        if arity == 1: return np.random.choice(self.func_ar1_list, p=self.prob_func_ar1)
        elif arity == 2: return np.random.choice(self.func_ar2_list, p=self.prob_func_ar2)
        else: return np.random.choice(self.func_list, p=self.prob_func)

    def gen_tree(self, depth=0):
        if depth<self.min_depth:
            if depth==0:
                n1 = GPTree(self.random_func(arity=2))
            else:
                n1 = GPTree(self.random_func())
        elif depth>=self.max_depth:
            n1 = GPTree(self.random_terminal())
        else:
            if random.random()<self.prob_node_func:
                n1 = GPTree(self.random_func())
            else:
                n1 = GPTree(self.random_terminal())
        if n1.is_func():
            n1.left=self.gen_tree(depth+1)
            if n1.arity()==2:
                n1.right=self.gen_tree(depth+1)
        return n1


    def fill_K_best(self):
        return self.P[np.argpartition(self.fitness, self.K_elite)[:self.K_elite]]


    def tournament(self, N):
        winners = []
        for i in range(N):		
            idx = np.random.randint(0, self.M, size=self.K_tourn)
            winner=self.P[np.argmin(self.fitness[idx])]
            winners.append(winner)
        return np.fromiter(winners, dtype=GPTree)


    def crossover(self, tree1, tree2):
        tree1, tree2 = tree1.clone(), tree2.clone()

        node1=random.choice(list(tree1)[1:])
        node2=random.choice(list(tree2)[1:])
        
        node1.val,node2.val=node2.val,node1.val
        node1.right,node2.right=node2.right,node1.right
        node1.left,node2.left=node2.left,node1.left
        return [tree1, tree2]


    def crossover_single(self, tree1):
        tree1 = tree1.clone()
        tree2 = self.tournament(1)[0].clone()

        node1=random.choice(list(tree1)[1:])
        node2=random.choice(list(tree2)[1:])
        
        node1.val=node2.val
        node1.right=node2.right
        node1.left=node2.left
        return tree1


    def mutation_tree(self, tree):
        tree = tree.clone()
        node_parent=random.choice(list(tree)[1:])
        node_parent.val=self.random_func()
    
        node_parent.left=self.gen_tree(random.randint(self.min_depth+1,self.max_depth))
        if node_parent.arity()==2:
            node_parent.right=self.gen_tree(random.randint(self.min_depth+1,self.max_depth))
        else:
            node_parent.right=None
        return tree
    

    def mutation_element(self, tree):
        tree = tree.clone()
        node=random.choice(list(tree)[0:])
        if node.is_func(): node.val=self.random_func(arity=node.arity())
        else: node.val=self.random_terminal()
        return tree

    def tree_operate(self, tree):
        tree_op = np.random.choice(
            [self.crossover_single, self.mutation_tree, self.mutation_element, GPTree.clone], 
            p=self.probs_tree_op)
        return tree_op(tree)

    def eval_fitness(self, tree, x, y, w=None):
        y_pred = tree.calculate_recursive(x)
        return self.error_fun(y,y_pred,w=w)*(1+tree.depth()*self.depth_penalty)
    
    def update_stats(self):
        best_ind = np.argmin(self.fitness)
        self.best_fitness.append(self.fitness[best_ind])
        self.best_trees.append(self.P[best_ind])
        self.mean_fitness.append(self.fitness.mean())

        depths = np.asarray([tree.depth() for tree in self.P])
        self.P_max_depth.append(depths.max())
        self.P_mean_depth.append(depths.mean())

    def get_stats(self):
        return {'total_generations': self.total_generations,
                'best_fitness': np.asarray(self.best_fitness),
                'best_trees': np.fromiter(self.best_trees, dtype=GPTree),
                'mean_fitness': np.asarray(self.mean_fitness),
                'P_max_depth': np.asarray(self.P_max_depth),
                'P_mean_depth': np.asarray(self.P_mean_depth)
                }


    def execute(self, x, y, generations=100, progressbar = True, resume=False):
        eval_fitness_vec=np.vectorize(self.eval_fitness, excluded=('x','y','w'))
        if not resume or (self.P is None or self.fitness is None):
            self.P = np.fromiter([self.gen_tree() for i in range(self.M)], dtype=GPTree)
            self.fitness = eval_fitness_vec(self.P, x=x, y=y)
            self.update_stats()
        range_gen = tqdm(range(int(generations)), desc="Progress") if progressbar else range(int(generations))
        for i in range_gen:
            new_P = []
            P_elite = self.fill_K_best()
            new_P.extend(P_elite)
            P_tournament = self.tournament(N=self.M-self.K_elite)	
            new_P.extend(self.tree_operate(tree) for tree in P_tournament)

            self.P = np.fromiter(new_P, dtype=GPTree)
            self.fitness = eval_fitness_vec(self.P, x=x, y=y)
            self.update_stats()
        self.total_generations += generations
        return self.get_stats()






    