#!/bin/bash

n_jobs=16
time_limit=300
dataset=feynman
vs_method=BART_VIP

for step in 1 2 3 4 5; do
    echo "Running Sympy Step ${step} on ${dataset}_SR_${vs_method}..."
    nohup python analyze_symbolic_model.py ../results_${dataset}/SR_${vs_method}/ \
        -results ../results_${dataset}/SR_${vs_method}/ \
        -data_path ~/feynman_dataset/ \
        -n_jobs ${n_jobs} \
        -timeout ${time_limit} \
        -step ${step} \
        >"logs/sympy_step${step}_${dataset}_SR_${vs_method}.out" \
        2>"logs/sympy_step${step}_${dataset}_SR_${vs_method}.err" &

    # Wait for the background job to finish
    wait $!

    # Check the exit status of the nohup command
    if [ $? -gt 0 ]; then
        echo "Sympy Step ${step} on ${dataset}_SR_${vs_method} failed... exiting loop."
        break
    fi

    echo "Finished Sympy Step ${step} on ${dataset}_SR_${vs_method}..."
done