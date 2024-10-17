from ..OperonRegressor import complexity, model, est
from operon.sklearn import SymbolicRegressor
from .params._operonregressor import params

est.set_params(**params)
est.allowed_symbols = "add,sub,mul,div,exp,log,sin,cos,sqrt,square,constant,variable"

# double the evals
est.max_evaluations = 1000000
est.generations = 100000  # just large enough since we have an evaluation budget
est.time_limit = int(24 * 60 * 60)
