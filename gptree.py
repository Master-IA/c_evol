# Definicion de la clase de arbol
# Extiende la clase Node de binarytree para ajustarlo a prog generica

from binarytree import Node
from funcs import FUNC_DICT, FUNC_LIST, SYMPY_FUNC_CONVERTER
import random
import sympy

SYMBOL = 'x'

# Arbol que contiene en cada nodo una string (si es funcion o SYMBOL) o un valor numerico (si es constante)
class GPTree(Node):

    def __init__(self, value, left=None, right=None):
        super().__init__(value, left, right)

# Para distinguir funciones de terminales
    def is_func(self):
        return self.value in FUNC_LIST

# Permite llamar la funcion de un nodo directamente
# n1=GPtree('add') -> n1(1,2)=3
    def __call__(self, *args):
        if self.is_func():
            return FUNC_DICT[self.val](*args)
        elif self.val==SYMBOL:
            return args[0] # si es 'x' se sustituye por el 1er arg
        else:
            return self.val # si es constante se devuelve ella misma

# Comprueba la paridad de una funcion mirando en el dict
    def arity(self):
        return FUNC_DICT[self.val].arity if self.is_func() else 0

# Calcula el valor de un arbol recursivamente
# (Peligro de desborde de pila si mucha profundidad)
    def calculate_recursive(self, x):
        if self.is_func():
            return self(self.left.calculate_recursive(x), self.right.calculate_recursive(x)) if self.arity() == 2 \
                else self(self.left.calculate_recursive(x))
        else:
            return self(x)
    
# Calcula el valor de un arbol con un stack tras sacar el postorden
    def calculate_stack(self, x):
        stack = []
        for nod in self.postorder:
            if nod.is_func():
                stack.append(
                    nod(stack.pop(), stack.pop()) if nod.arity()==2 
                    else nod(stack.pop()))
            else:
                stack.append(nod(x))
        return stack[0]

# Devuelve el arbol como una funcion en una sola linea
# pej 1+5*(1/x) -> add (1,mul(5,inv(x)))
    def __str__(self):
        output, terminals = '', [0]
        for nod in self.preorder:
            if nod.is_func():
                output += (str(nod.val) + '(')
                terminals.append(nod.arity())
            else:
                output += str(nod.val)          
                terminals[-1] -= 1
                if terminals[-1] > 0: output += ','
                while terminals[-1] == 0:            
                    terminals.pop()
                    terminals[-1] -= 1                    
                    output += ')'
                    if terminals[-1] > 0: output += ','
        return output

# Devuelve un clon del arbol, mas eficiente y rapido que copy.deepcopy
    def clone(self):        
        other = GPTree(self.val)

        stack1 = [self]
        stack2 = [other]

        while stack1 or stack2:
            node1 = stack1.pop()
            node2 = stack2.pop()

            if node1.left is not None:
                node2.left = GPTree(node1.left.val)
                stack1.append(node1.left)
                stack2.append(node2.left)

            if node1.right is not None:
                node2.right = GPTree(node1.right.val)
                stack1.append(node1.right)
                stack2.append(node2.right)
        return other

# Devuelve la profundidad del arbol, 1 unico nodo -> depth=0
    def depth(self):
        max_leaf_depth = -1
        current_nodes = [self]

        while len(current_nodes) > 0:
            max_leaf_depth += 1
            next_nodes = []

            for node in current_nodes:
                if node.left is not None:                    
                    next_nodes.append(node.left)                
                if node.right is not None:                    
                    next_nodes.append(node.right)
            current_nodes = next_nodes

        return max_leaf_depth

# Devuelve un nodo aleatorio del arbol
# depth_weighted pondera en funcion de la prof (>prof -> >prob) y con first_depth se indica desde que prof tomar
# Si no, se toma uniforme, y con skip_root omite el nodo raiz
    def random_node(self, skip_root=False, first_depth = 1 ,depth_weighted=True):
        if depth_weighted:
            #first_depth = 2 if skip_root else 0
            tree_bydepth = self.levels[first_depth:]
            
            weights = range(2, len(tree_bydepth)+first_depth)
            
            node = random.choice(random.choices(tree_bydepth, weights=weights)[0])
        else:
            first = 1 if skip_root else 0
            node = random.choice(list(self)[first:])
        return node

# Devuelve el arbol en la simplificacion de sympy
    def sympify_str(self):
        return sympy.sympify(str(self),locals=SYMPY_FUNC_CONVERTER)

# Para sobrecargar la representacion de binarytree
    def _repr_svg_(self):
        return str(self)

    def graphviz(self, *args, **kwargs) :
        return super().graphviz(*args, **kwargs)

