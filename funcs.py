import numpy as np

INV_THRESHOLD = 0.001
LOG_THRESHOLD = 0.001

class Func():
	def __init__(self, operator, name, arity):
		self.operator = operator
		self.name = name
		self.arity = arity

	def __str__(self):
		return self.name

	def __call__(self, *args):
		return self.operator(*args)


def safe_inv(x1,x2=None):
	return x1 if np.abs(x1) <= INV_THRESHOLD else 1/x1

def safe_log(x1,x2=None):
	return x1 if np.log(x1) <= LOG_THRESHOLD else 0

# mejor usar un diccionario, no se puede indexar un enum por numero cuando no tienen valores numericos
"""class Func_Enum(Enum):
	ADD = Func(np.add, "add", 2)
	SUB = Func(np.subtract, "sub", 2)
	MUL = Func(np.multiply, "mul", 2)
	INV = Func(safe_inv, "inv", 1)
	LOG = Func(safe_log, "log", 1)
ar1_list = [Func_Enum.INV, Func_Enum.LOG]
ar2_list = [Func_Enum.ADD, Func_Enum.SUB, Func_Enum.MUL]
"""

FUNC_DICT = {
	'add' : Func(np.add, "add", 2),
	'sub' : Func(np.subtract, "sub", 2),
	'mul' : Func(np.multiply, "mul", 2),
	'inv' : Func(safe_inv, "inv", 1),
	'log' : Func(safe_log, "log", 1)
}