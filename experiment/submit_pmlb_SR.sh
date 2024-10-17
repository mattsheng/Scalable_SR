#!/bin/bash

for ml in "tuned.OperonRegressor"; do
    echo "Running ${ml} on PMLB datasets..."
    nohup python analyze.py /path/to/pmlb \
        -ml ${ml} \
        -results ../results_blackbox/SR/ \
        -n_trials 10 \
        -n_jobs 1 \
        -time_limit 24:00 \
        -tuned -skip_tuning \
        >"logs/pmlb_${ml}.out" \
        2>"logs/pmlb_${ml}.err" &

    # Wait for the background job to finish
    wait $!

    # Check the exit status of the nohup command
    if [ $? -gt 0 ]; then
        echo "Job with ${ml} failed, exiting loop."
        break
    fi

    echo "Finished ${ml}..."
done