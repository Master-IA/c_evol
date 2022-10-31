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

# para que tree.calculate_recursive(x) siempre devuelva vector de len(x)
# ya que cuando es una constante numpy lo simplifica a un solo float
def _to_vector(vec, size):
    if vec.shape==(): return np.resize(vec,size)
    else: return vec

# para normalizar pesos a probs que sumen 1
def _normalize_probs(probs):
    probs=np.asarray(probs)
    return probs/sum(probs)

class GP():
    def __init__(self, 
                M=500,                          # num de individuos en poblacion 
                tourn=0.03,                     # % de M que se toma en cada torneo
                elitism=0.1,                    # % de M que se selecciona por elitismo
                const_range=(0,2),             # rango uniforme del que se toman terminales constantes
                min_depth=2,                    # minima prof de crecimiento para arboles y subarboles, en prof previas se toman solo funciones
                max_depth=10,                   # max prof de crecimiento para arboles y subarboles, cuando se alcanza se toman solo terminales
                prob_node_symb=0.5,             # probabilidad de que se asigne el simbolo al crear un nodo terminal
                prob_node_func=0.5,             # probabilidad de que se asigne una funcion (y no un terminal) al crear un nodo cualquiera
                func_list=FUNC_LIST,            # lista (de strings) de las funciones a elegir para los nodos
                prob_func=None,                 # lista de probabilidades para las funciones, debe sumar 1 (las probs para cada aridad se ajustan de forma condicionada)
                probs_tree_op=[0.9,0.01,0.01],  # lista de probs para operaciones post seleccion (cruce, mut. arbol, mut.puntual), si tiene long 3 se ajusta la prob de inaccion
                error_fun=rmse,                 # funcion de calculo de error (fitness a minimizar), toma vector de y, vector de y_pred
                depth_penalty=0,                # penalizacion en fitness por profundidad del arbol, aumenta fitness en un prof*penalizacion% (tomar valores < 0.001)
                calc_weights=False              # ponderar fitness punto a punto en funcion de ajuste global (NO USAR)
                ):

        self.M = M
        self.tourn, self.K_tourn = tourn, max([1,int(M*tourn)]) # si la poblacion no permite ese % de torneo se coge solo 1
        self.elitism, self.K_elite = elitism, int(M*elitism)        
        self.const_range = const_range
        self.min_depth, self.max_depth = min_depth, max_depth
        self.prob_node_symb, self.prob_node_func = prob_node_symb, prob_node_func        

        self.func_list = func_list
        self.prob_func = _normalize_probs(prob_func) if prob_func is not None else _normalize_probs(np.ones(len(func_list)))
        # la lista de funciones se distingue a su vez por la aridad y se calculan las probs
        ar1_mask, ar2_mask = np.isin(self.func_list,FUNC_AR1_LIST), np.isin(self.func_list,FUNC_AR2_LIST)
        self.func_ar1_list, self.func_ar2_list = np.asarray(self.func_list)[ar1_mask], np.asarray(self.func_list)[ar2_mask]
        self.prob_func_ar1, self.prob_func_ar2 = _normalize_probs(np.asarray(self.prob_func)[ar1_mask]), _normalize_probs(np.asarray(self.prob_func)[ar2_mask])


        self.probs_tree_op = probs_tree_op
        # si no se incluye se incorpora la prob de no hacer nada como 1 - la suma del resto
        if len(probs_tree_op)<4: self.probs_tree_op.append(1-sum(probs_tree_op))

        self.error_fun, self.depth_penalty = error_fun, depth_penalty
        self.calc_weights, self.weights = calc_weights, None

        
        self.P, self.fitness, self.fitness_p = None, None, None
        self.total_generations = 0
        self.best_fitness, self.best_fitness_p, self.mean_fitness, self.mean_fitness_p = [], [], [], []
        self.best_trees, self.best_trees_p = [], []
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

    def gen_tree(self, depth=0, root=False, root_right=False):
        if depth<self.min_depth:
            if depth==0:
                if root==True:
                    n1 = GPTree(FUNC_AR2_LIST[3])
                else:
                    n1 = GPTree(self.random_func(arity=2))
            
            elif root_right==True:
                n1 = GPTree(FUNC_AR1_LIST[1])
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
            if root==True:
                n1.left = GPTree(SYMBOL)
            else:
                n1.left=self.gen_tree(depth+1, False, False)
            if n1.arity()==2:
                if root==True:
                    n1.right=self.gen_tree(depth+1,False, True)
                else:    
                    n1.right=self.gen_tree(depth+1,False, False)
        return n1


    def fill_K_best(self):
        return self.P[np.argpartition(self.fitness_p, self.K_elite)[:self.K_elite]]


    def tournament(self, N):
        winners = []
        for i in range(N):		
            idx = np.random.randint(0, self.M, size=self.K_tourn)
            winner=self.P[np.argmin(self.fitness_p[idx])]
            winners.append(winner)
        return np.fromiter(winners, dtype=GPTree)



    def crossover(self, tree1, tree2):
        tree1, tree2 = tree1.clone(), tree2.clone()

        node1=tree1.random_node(skip_root=True, first_depth = 2 ,depth_weighted=True) #random.choice(list(tree1)[1:])
        node2=tree2.random_node(skip_root=True, first_depth = 2 ,depth_weighted=True) #random.choice(list(tree2)[1:])
        
        node1.val,node2.val=node2.val,node1.val
        node1.right,node2.right=node2.right,node1.right
        node1.left,node2.left=node2.left,node1.left
        return [tree1, tree2]


    def crossover_single(self, tree1):
        tree1 = tree1.clone()
        tree2 = self.tournament(1)[0].clone()

        node1=tree1.random_node(skip_root=True, first_depth = 2 ,depth_weighted=True) #random.choice(list(tree1)[1:])
        node2=tree2.random_node(skip_root=True, first_depth = 2 ,depth_weighted=True) #random.choice(list(tree2)[1:])
        
        node1.val=node2.val
        node1.right=node2.right
        node1.left=node2.left
        return tree1


    def mutation_tree(self, tree):
        tree = tree.clone()
        node_parent=tree.random_node(skip_root=True, first_depth = 2 ,depth_weighted=True) #random.choice(list(tree)[1:])
        node_parent.val=self.random_func()
    
        node_parent.left=self.gen_tree(random.randint(self.min_depth+1,self.max_depth), False, False)
        if node_parent.arity()==2:
            node_parent.right=self.gen_tree(random.randint(self.min_depth+1,self.max_depth), False, False)
        else:
            node_parent.right=None
        return tree
    

    def mutation_element(self, tree):
        tree = tree.clone()
        node=tree.random_node(skip_root=True, first_depth = 2 ,depth_weighted=True) #random.choice(list(tree)[0:])
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

    def eval_all_fitness(self, x, y):
        N = len(x)
        # matriz MxN de predicciones por cada arbol de cada y
        y_preds = np.array([_to_vector(tree.calculate_recursive(x),N) for tree in self.P])
        if self.calc_weights: # pesos -> calculo error de todos los arboles para cada y[i] individual
            self.weights = np.asarray([self.error_fun(y_preds[:,i],y[i]) for i in range(len(y))])
        # fitness -> calculo error arbol a arbol para todos los y, penalizando profundidad en un depth*penalty % adicional
        fitness = np.apply_along_axis(self.error_fun,1, y_preds, y, self.weights)
        fitness_p = np.apply_along_axis(self.error_fun,1, y_preds, y, self.weights)*(1+np.vectorize(GPTree.depth)(self.P)*self.depth_penalty)
        return fitness, fitness_p

    def update_stats(self):
        best_ind = np.argmin(self.fitness)        
        self.best_trees.append(self.P[best_ind])
        self.best_fitness.append(self.fitness[best_ind])
        self.mean_fitness.append(self.fitness.mean())

        best_ind_p = np.argmin(self.fitness_p)  
        self.best_trees_p.append(self.P[best_ind_p])
        self.best_fitness_p.append(self.fitness_p[best_ind_p])
        self.mean_fitness_p.append(self.fitness_p.mean())

        depths = np.asarray([tree.depth() for tree in self.P])
        self.P_max_depth.append(depths.max())
        self.P_mean_depth.append(depths.mean())

    def get_stats(self):
        return {'total_generations': self.total_generations,
                'best_fitness': np.asarray(self.best_fitness),
                'best_trees': np.fromiter(self.best_trees, dtype=GPTree),
                'mean_fitness': np.asarray(self.mean_fitness),
                'best_fitness_p': np.asarray(self.best_fitness_p),
                'mean_fitness_p': np.asarray(self.mean_fitness_p),
                'P_max_depth': np.asarray(self.P_max_depth),
                'P_mean_depth': np.asarray(self.P_mean_depth)
                }


    def execute(self, x, y, generations=100, progressbar = True, resume=False):
        if not resume or (self.P is None or self.fitness is None):
            self.P = np.fromiter([self.gen_tree(root=True) for i in range(self.M)], dtype=GPTree)
            self.fitness, self.fitness_p = self.eval_all_fitness(x,y) 
            self.update_stats()
        range_gen = tqdm(range(int(generations)), desc="Progress") if progressbar else range(int(generations))

        for _ in range_gen:
            new_P = []
            P_elite = self.fill_K_best()
            new_P.extend(P_elite)
            P_tournament = self.tournament(N=self.M-self.K_elite)	
            new_P.extend(self.tree_operate(tree) for tree in P_tournament)

            self.P = np.fromiter(new_P, dtype=GPTree)
            self.fitness , self.fitness_p= self.eval_all_fitness(x,y)
            self.update_stats()
        self.total_generations += generations
        return self.get_stats()


