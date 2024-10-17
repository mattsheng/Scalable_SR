from ..FE_AFPRegressor import complexity, est, eval_kwargs, model, pre_train
from .params._fe_afpregressor import params

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
est.time_limit = int(24 * 60 * 60)
# doubling evals
est.g = int(est.g * 2**0.5)
est.popsize = int(est.popsize * 2**0.5)
