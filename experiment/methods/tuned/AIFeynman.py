from ..AIFeynman import complexity, est, model
from .params._aifeynman import params

est.set_params(**params)
est.max_time = int(24 * 60 * 60)
