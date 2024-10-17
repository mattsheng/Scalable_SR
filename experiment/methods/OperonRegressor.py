from operon.sklearn import SymbolicRegressor

est = SymbolicRegressor(
            local_iterations=5,
            generations=10000, # just large enough since we have an evaluation budget
            n_threads=1,
            random_state=None,
            time_limit=int(23.5*60*60), # 23.5 hours
            max_evaluations=int(5e5),
            population_size=500
            )

hyper_params = [
        {
            'population_size': (100,),
            'pool_size': (100,),
            'max_length': (25,),
            'allowed_symbols': ('add,mul,aq,constant,variable',),
            'local_iterations': (5,),
            'offspring_generator': ('basic',),
            'tournament_size': (3,),
            'reinserter': ('keep-best',),
            'max_evaluations': (int(5e5),)
        },
        {
            'population_size': (100,),
            'pool_size': (100,),
            'max_length': (25,),
            'allowed_symbols': ('add,mul,aq,exp,log,sin,tanh,constant,variable',),
            'local_iterations': (5,),
            'offspring_generator': ('basic',),
            'tournament_size': (3,),
            'reinserter': ('keep-best',),
            'max_evaluations': (int(5e5),)
        },
        {
            'population_size': (100,),
            'pool_size': (100,),
            'max_length': (50,),
            'allowed_symbols': ('add,mul,aq,constant,variable',),
            'local_iterations': (5,),
            'offspring_generator': ('basic',),
            'tournament_size': (3,),
            'reinserter': ('keep-best',),
            'max_evaluations': (int(5e5),)
        },
        {
            'population_size': (500,),
            'pool_size': (500,),
            'max_length': (25,),
            'allowed_symbols': ('add,mul,aq,constant,variable',),
            'local_iterations': (5,),
            'offspring_generator': ('basic',),
            'tournament_size': (5,),
            'reinserter': ('keep-best',),
            'max_evaluations': (int(5e5),)
        },
        {
            'population_size': (500,),
            'pool_size': (500,),
            'max_length': (25,),
            'allowed_symbols': ('add,mul,aq,exp,log,sin,tanh,constant,variable',),
            'local_iterations': (5,),
            'offspring_generator': ('basic',),
            'tournament_size': (5,),
            'reinserter': ('keep-best',),
            'max_evaluations': (int(5e5),)
        },
        {
            'population_size': (500,),
            'pool_size': (500,),
            'max_length': (50,),
            'allowed_symbols': ('add,mul,aq,constant,variable',),
            'local_iterations': (5,),
            'offspring_generator': ('basic',),
            'tournament_size': (5,),
            'reinserter': ('keep-best',),
            'max_evaluations': (int(5e5),)
        },
    ]

def complexity(est):
    return est.stats_['model_complexity'] # scaling nodes not counted

def model(est, X):
    return est.get_model_string(3)
