from ellyn import ellyn

# 500,000 evaluations = 250,000 with 1 constant hill climbing iteration
pop_sizes = [100, 500, 1000]
gs = [2500, 500, 250]
op_lists=[
        ['n','v','+','-','*','/','sin','cos','exp','log','2','3', 'sqrt'],
        ['n','v','+','-','*','/', 'exp','log','2','3', 'sqrt']
        ]

hyper_params = []

for p, g in zip(pop_sizes, gs):
    for op_list in op_lists:
        hyper_params.append({
                'popsize':[p],
                'g':[g],
                'op_list':[op_list]
                })


# Create the pipeline for the model
est = ellyn(selection='epsilon_lexicase',
            lex_eps_global=False,
            lex_eps_dynamic=False,
            islands=False,
            num_islands=10,
            island_gens=100,
            verbosity=0,
            print_data=False,
            elitism=True,
            pHC_on=True,
            prto_arch_on=True,
            max_len = 64,
            max_len_init=20,
            popsize=500,
            g=500,
            time_limit=int(23.5*60*60)
            )

def complexity(est):
    return len(est.best_estimator_)

def model(est):
    return est.stack_2_eqn(est.best_estimator_)
