from ..MRGPRegressor import complexity, est, model
from ..src.mrgp import MRGPRegressor
from .params._mrgpregressor import params

est.set_params(**params)

# double the evals
est.time_out = int(24 * 60 * 60)
est.g = int(est.g * 2**0.5)
est.popsize = int(est.popsize * 2**0.5)
