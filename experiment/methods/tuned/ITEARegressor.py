from ..ITEARegressor import complexity, est, eval_kwargs, model
from .params._itearegressor import params

est.set_params(**params)
est.transfunctions = "[Id, Tanh, Sin, Cos, Log, Exp, SqrtAbs]"

# double the evals
est.npop = int(est.npop * 2**0.5)
est.ngens = int(est.ngens * 2**0.5)
