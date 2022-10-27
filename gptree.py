from binarytree import Node
from funcs import FUNC_DICT, FUNC_LIST, SYMPY_FUNC_CONVERTER
import sympy

INV_THRESHOLD = 0.001
LOG_THRESHOLD = 0.001

SYMBOL = 'x'

class GPTree(Node):
    def __init__(self, value, left=None, right=None):
        super().__init__(value, left, right)

    def is_func(self):
        return self.value in FUNC_LIST

# permite llamar la funcion de un nodo directamente
# n1=GPtree('add') -> n1(1,2)=3
    def __call__(self, *args):
        if self.is_func():
            return FUNC_DICT[self.val](*args)
        elif self.val==SYMBOL:
            return args[0] # si es 'x' se sustituye por el 1er arg
        else:
            return self.val # si es constante se devuelve ella misma

# comprueba la paridad de una funcion mirando en el dict
# se podria poner como atributo y no como funcion, pero + gasto memoria por nodo
    def arity(self):
        return FUNC_DICT[self.val].arity if self.is_func() else 0

# calcula el valor de un arbol recursivamente
# peligro de desborde de pila si mucha profundidad
    def calculate_recursive(self, x):
        if self.is_func():
            return self(self.left.calculate_recursive(x), self.right.calculate_recursive(x)) if self.arity() == 2 \
                else self(self.left.calculate_recursive(x))
        else:
            return self(x)
    
# calcula el valor de un arbol con un stack tras sacar el postorden
# otra alternativa seria copiar alg de postorden de API y asi no iterar 2 veces
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

# devuelve el arbol como una funcion de Lisp
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

# Devuelve el arbol en la simplificacion de sympy
    def sympify_str(self):
        return sympy.sympify(str(self),locals=SYMPY_FUNC_CONVERTER)
    
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