from pysr import PySRRegressor


def complexity(est):
    return est.get_best()["complexity"]


def model(est):
    return str(est.sympy())


est = PySRRegressor(
    early_stop_condition=(
        "stop_if(loss, complexity) = loss < 1e-4 && complexity < 20"
        # Stop early if we find a good and simple equation
    ),
    niterations=1_000_000_000,
    # ncyclesperiteration=2_500,
    population_size=100,
    populations=15,
    timeout_in_seconds=int(24 * 60 * 60),
    maxsize=30,
    # maxdepth=20,
    binary_operators=["+", "-", "*", "/"],
    unary_operators=["sin", "exp", "log", "sqrt", "square", "abs"],
    constraints={
        **dict(
            sin=9,
            exp=9,
            log=9,
            sqrt=9,
            square=9,
        ),
        **{"/": (-1, 9)},
    },
    nested_constraints=dict(
        sin=dict(
            sin=0,
            exp=1,
            log=1,
            sqrt=1,
        ),
        exp=dict(
            exp=0,
            log=0,
        ),
        log=dict(
            exp=0,
            log=0,
        ),
        sqrt=dict(
            sqrt=0,
            square=0,
        ),
        abs=dict(
            square=0,
            sqrt=0,
            log=0,
        ),
    ),
    # prefer multiprocessing:
    procs=0,
    multithreading=False,
    # batching=True,
    # batch_size=50,
    turbo=True,
    weight_optimize=0.001,
    adaptive_parsimony_scaling=1000.0,
    parsimony=0.0,
    temp_equation_file=True,
    delete_tempfiles=True,
    cluster_manager="none",
    verbosity=0,
    progress=False,
    print_precision=2,
)

# See https://astroautomata.com/PySR/tuning/ for tuning advice
hyper_params = [{}]
