#!/bin/bash

vs_method=BART_VIP
rep=20
for ml in "tuned.OperonRegressor"; do
    echo "Running ${ml}+${vs_method} on PMLB datasets with rep = ${rep}..."
    nohup python analyze.py /path/to/pmlb \
        -ml ${ml} \
        -results ../results_blackbox/SR_${vs_method}/ \
        -n_trials 10 \
        -n_jobs 1 \
        -time_limit 24:00 \
        -vs_method ${vs_method} \
        -vs_result_path ../results_blackbox/pmlb_${vs_method}_withidx.feather \
        -vs_idx_label idx_hclst \
        -rep ${rep} \
        -tuned -skip_tuning \
        >"logs/pmlb_${ml}_${vs_method}_${rep}.out" \
        2>"logs/pmlb_${ml}_${vs_method}_${rep}.err" &

    # Wait for the background job to finish
    wait $!

    # Check the exit status of the nohup command
    if [ $? -gt 0 ]; then
        echo "Job with ${ml}+${vs_method} failed, exiting loop."
        break
    fi

    echo "Finished ${ml}+${vs_method}..."
done