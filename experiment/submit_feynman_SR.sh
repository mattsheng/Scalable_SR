#!/bin/bash

for ml in "tuned.OperonRegressor"; do
    for n in 500 1000 1500 2000; do
        for snr in 0 20 15 10 5 2 1 0.5; do
            echo "Running ${ml} on Feynman datasets with n = ${n}, SNR = ${snr},..."
            nohup python analyze.py /path/to/feynman \
                -ml ${ml} \
                -results ../results_feynman/SR/n_${n}/ \
                -signal_to_noise $snr \
                -n $n \
                -sym_data \
                -n_trials 10 \
                -n_jobs 1 \
                -time_limit 24:00 \
                -tuned -skip_tuning \
                >"logs/feynman_${ml}_n${n}_snr${snr}.out" \
                2>"logs/feynman_${ml}_n${n}_snr${snr}.err" &

            # Wait for the background job to finish
            wait $!

            # Check the exit status of the nohup command
            if [ $? -gt 0 ]; then
                echo "Job with ${ml}, n = ${n}, SNR = ${snr} failed, exiting loop."
                break
            fi

            echo "Finished ${ml}, n = ${n}, SNR = ${snr}..."
        done
    done
done