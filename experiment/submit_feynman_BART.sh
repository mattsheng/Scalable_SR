#!/bin/bash

vs_method=BART_VIP
rep=20
for n in 500 1000 1500 2000; do
    for snr in 0 20 15 10 5 2 1 0.5; do
        echo "Running ${vs_method} on Feynman datasets with n = ${n}, SNR = ${snr}, rep = ${rep}..."
        nohup python analyze.py /path/to/feynman \
            -script BART_selection \
            -ml ${vs_method} \
            -results ../results_feynman/${vs_method}/n_${n}/ \
            -signal_to_noise $snr \
            -n $n \
            -sym_data \
            -n_trials 10 \
            -n_jobs 1 \
            -rep $rep \
            -time_limit 24:00 \
            >"logs/feynman_${vs_method}_n${n}_snr${snr}_rep${rep}.out" \
            2>"logs/feynman_${vs_method}_n${n}_snr${snr}_rep${rep}.err" &

        # Wait for the background job to finish
        wait $!

        # Check the exit status of the nohup command
        if [ $? -gt 0 ]; then
            echo "Job with ${vs_method}, n = ${n}, SNR = ${snr}, rep = ${rep} failed, exiting loop."
            break
        fi

        echo "Finished ${vs_method}, n = ${n}, SNR = ${snr}, rep = ${rep}..."
    done
done