from ..EPLEXRegressor import complexity, est, model
from .params._eplexregressor import params

est.set_params(**params)
est.op_list = [
    "n",
    "v",
    "+",
    "-",
    "*",
    "/",
    "exp",
    "log",
    "2",
    "3",
    "sqrt",
    "sin",
    "cos",
]

# double the evals
est.g = int(est.g * 2**0.5)
est.popsize = int(est.popsize * 2**0.5)
est.time_limit = int(24 * 60 * 60)
