# Definicion de las funciones para el arbol
# Contiene la clase Func, el diccionario de funciones y las definiciones seguras de las que lo necesitan

import numpy as np

INV_THRESHOLD = 0.001
LOG_THRESHOLD = 0.001
SQRT_THRESHOLD = 0.001
EXP_THRESHOLD = 0.001

# Clase de funciones, todas tienen operador, nombre y aridad
class Func():
	def __init__(self, operator, name, arity):
		self.operator = operator
		self.name = name
		self.arity = arity

	def __str__(self):
		return self.name

	def __call__(self, *args):
		return self.operator(*args)

# Se protege el inverso devolviendo x si es menor absolutamente que el umbral
def safe_inv(x1,x2=None):
	with np.errstate(divide='ignore', invalid='ignore'):
		return np.where(np.abs(x1) > INV_THRESHOLD, 1/x1, x1)

# Se protege el logaritmo devolviendo 0 si es menor absolutamente con el umbral
# Para permitir valores negativos, se toma el logaritmo del absoluto
def safe_log(x1,x2=None):
	with np.errstate(divide='ignore', invalid='ignore'):
		return np.where(np.abs(x1) > LOG_THRESHOLD, np.log(np.abs(x1)), 0)

# Se protege la raiz cuadrada devolviendo 0 si es menor absolutamente que el umbral
# Para permitir valores negativos, se toma la raiz del absoluto
def safe_sqrt(x1,x2=None):
	with np.errstate(divide='ignore', invalid='ignore'):
		return np.where(np.abs(x1) > SQRT_THRESHOLD, np.sqrt(np.abs(x1)), 0)

# Se protege la division devolviendo x1 si x2 es menor absolutamente que el umbral
def safe_div(x1,x2):
	with np.errstate(divide='ignore', invalid='ignore'):
		return np.where(np.abs(x2) > INV_THRESHOLD, x1/x2, x1)

# Se protege la exponencial devolviendo 0 si es menor absolutamente que el umbral
# Se toma siempre la exponencial del absoluto
def safe_exp(x1,x2=None):
	with np.errstate(divide='ignore', invalid='ignore'):
			return np.where(np.abs(x1) > EXP_THRESHOLD, np.exp(np.abs(x1)), 0)

# Separamos las funciones en listas por su aridad
FUNC_AR1_LIST = ['inv', 'log','sqrt', 'exp', 'floor']	
FUNC_AR2_LIST = ['add', 'sub', 'mul', 'div', 'max', 'min']
FUNC_LIST = FUNC_AR1_LIST + FUNC_AR2_LIST

# Sus operadores se encuentran indexados por nombre en un diccionario de funciones
FUNC_DICT = {
	'add' : Func(np.add, "add", 2),
	'sub' : Func(np.subtract, "sub", 2),
	'mul' : Func(np.multiply, "mul", 2),
	'div' : Func(safe_div, "div", 2),
	'inv' : Func(safe_inv, "inv", 1),
	'log' : Func(safe_log, "log", 1),
	'max' : Func(np.maximum, "max", 2),
	'min' : Func(np.minimum, "min", 2),
	'sqrt' : Func(safe_sqrt, "sqrt", 1),
	'exp': Func(safe_exp, "exp", 1),
	'floor': Func(np.floor, "floor", 1)
}

# Para usarse la simplificacion de simpy
SYMPY_FUNC_CONVERTER = {
    'add': lambda x, y : x + y,
    'sub': lambda x, y : x - y,
    'mul': lambda x, y : x*y,
    #'div': lambda x, y : x/y,
    #'sqrt': lambda x : x**0.5,
    #'log': lambda x : math.log(x),
    'abs': lambda x : abs(x),
    'neg': lambda x : -x,
    #'inv': lambda x : 1/x,
    #'sin': lambda x : math.sin(x),
    #'cos': lambda x : math.cos(x),
}