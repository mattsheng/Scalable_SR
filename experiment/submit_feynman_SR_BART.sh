#!/bin/bash

vs_method=BART_VIP
for ml in "tuned.OperonRegressor"; do
    for n in 500 1000 1500 2000; do
        for snr in 0 20 15 10 5 2 1 0.5; do
            echo "Running ${ml}+${vs_method} on Feynman datasets with n = ${n}, SNR = ${snr},..."
            nohup python analyze.py /path/to/feynman \
                -ml ${ml} \
                -results ../results_feynman/SR_${vs_method}/n_${n}/ \
                -signal_to_noise $snr \
                -n $n \
                -sym_data \
                -n_trials 10 \
                -n_jobs 1 \
                -time_limit 24:00 \
                -vs_method ${vs_method} \
                -vs_result_path ../results_feynman/feynman_${vs_method}_withidx.feather \
                -vs_idx_label idx_hclst \
                -rep 20 \
                -tuned -skip_tuning \
                >"logs/feynman_${ml}_${vs_method}_n${n}_snr${snr}.out" \
                2>"logs/feynman_${ml}_${vs_method}_n${n}_snr${snr}.err" &

            # Wait for the background job to finish
            wait $!

            # Check the exit status of the nohup command
            if [ $? -gt 0 ]; then
                echo "Job with ${ml}+${vs_method}, n = ${n}, SNR = ${snr} failed, exiting loop."
                break
            fi

            echo "Finished ${ml}+${vs_method}, n = ${n}, SNR = ${snr}..."
        done
    done
done