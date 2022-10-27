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
	with np.errstate(divide='ignore', invalid='ignore'):
		return np.where(np.abs(x1) > INV_THRESHOLD, 1/x1, x1)

def safe_log(x1,x2=None):
	with np.errstate(divide='ignore', invalid='ignore'):
		return np.where(np.abs(x1) > LOG_THRESHOLD, np.log(np.abs(x1)), 0)

FUNC_AR1_LIST = ['inv', 'log']
FUNC_AR2_LIST = ['add', 'sub', 'mul']
FUNC_LIST = FUNC_AR1_LIST + FUNC_AR2_LIST

FUNC_DICT = {
	'add' : Func(np.add, "add", 2),
	'sub' : Func(np.subtract, "sub", 2),
	'mul' : Func(np.multiply, "mul", 2),
	'inv' : Func(safe_inv, "inv", 1),
	'log' : Func(safe_log, "log", 1)
}