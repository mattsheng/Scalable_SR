#!/bin/bash

vs_method=BART_VIP
rep=20
echo "Running ${vs_method} on PMLB datasets with rep = ${rep}..."
nohup python analyze.py /path/to/pmlb \
    -script BART_selection \
    -ml ${vs_method} \
    -results ../results_blackbox/${vs_method}/ \
    -n_trials 10 \
    -n_jobs 1 \
    -rep $rep \
    -time_limit 24:00 \
    >"logs/pmlb_${vs_method}_${rep}.out" \
    2>"logs/pmlb_${vs_method}_${rep}.err" &

# Wait for the background job to finish
wait $!

# Check the exit status of the nohup command
if [ $? -gt 0 ]; then
    echo "Job with ${vs_method}, rep = ${rep} failed, exiting loop."
    break
fi

echo "Finished ${vs_method}, rep = ${rep}..."
