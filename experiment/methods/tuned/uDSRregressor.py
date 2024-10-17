import os

from dso import ParallelizedUnifiedDeepSymbolicRegressor

est = ParallelizedUnifiedDeepSymbolicRegressor(
    os.path.join(os.path.dirname(__file__), "../../config_regression.json")
)


def model(est, X=None):
    return str(est.expr)


def complexity(est):
    return est.complexity
