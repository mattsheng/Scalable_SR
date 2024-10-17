import json
import os
import subprocess
import sys
import time
from math import ceil

import numpy as np
import pandas as pd
import rpy2.robjects as robjects
from read_file import read_file
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from symbolic_utils import get_sym_model
from utils import jsonify


def set_env_vars(n_jobs):
    os.environ["OMP_NUM_THREADS"] = n_jobs
    os.environ["OPENBLAS_NUM_THREADS"] = n_jobs
    os.environ["MKL_NUM_THREADS"] = n_jobs


def run_BART(
    dataset,
    results_path,
    random_state,
    est_name,
    target_noise=0.0,
    feature_noise=0.0,
    signal_to_noise=0.0,
    # n_samples=10000,  # For blackbox datasets
    n_samples=0,  # For ground-truth datasets
    rep=0,
    scale_x=True,
    scale_y=True,
    sym_data=False,
    og_data=False,
    return_results=False,
):
    print(40 * "=", "Running BART variable selection on ", dataset, 40 * "=", sep="\n")

    np.random.seed(random_state)

    ##################################################
    # setup data
    ##################################################
    features, labels, feature_names = read_file(dataset)
    if sym_data:
        true_model = get_sym_model(dataset)
        p = features.shape[1]
        p0 = p - sum(1 for c in feature_names if "x_bad" in c)
        if og_data:
            features = features[:, :p0]
            feature_names = feature_names[range(p0)]

    # add noise according to SNR = Var(Y) / Var(eps)
    # i.e. SD(eps) = SD(Y) / sqrt(SNR)
    if signal_to_noise > 0:
        print("setting SNR to", signal_to_noise)
        labels += np.random.normal(
            loc=0,
            scale=np.std(labels) / np.sqrt(signal_to_noise),
            size=len(labels),
        )

    # generate train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, train_size=0.75, test_size=0.25, random_state=random_state
    )

    # if dataset is large, subsample the training set
    if n_samples > 0 and len(labels) > n_samples:
        print("subsampling training data from", len(X_train), "to", n_samples)
        sample_idx_train = np.random.choice(
            np.arange(len(X_train)), size=n_samples, replace=False
        )
        X_train = X_train[sample_idx_train]
        y_train = y_train[sample_idx_train]

        n_samples_test = ceil(n_samples / 3)
        print("subsampling testing data from", len(X_test), "to", n_samples_test)
        sample_idx_test = np.random.choice(
            np.arange(len(X_test)), size=n_samples_test, replace=False
        )
        X_test = X_test[sample_idx_test]
        y_test = y_test[sample_idx_test]

    # scale and normalize the data
    if scale_x:
        print("scaling X")
        sc_X = StandardScaler()
        X_train_scaled = sc_X.fit_transform(X_train)
    else:
        X_train_scaled = X_train

    if scale_y:
        print("scaling y")
        sc_y = StandardScaler()
        y_train_scaled = sc_y.fit_transform(y_train.reshape(-1, 1)).flatten()
    else:
        y_train_scaled = y_train

    # add noise to the target
    if target_noise > 0:
        print("adding", target_noise, "noise to target")
        y_train_scaled += np.random.normal(
            0,
            target_noise * np.sqrt(np.mean(np.square(y_train_scaled))),
            size=len(y_train_scaled),
        )
    # add noise to the features
    if feature_noise > 0:
        print("adding", target_noise, "noise to features")
        X_train_scaled = np.array(
            [
                x
                + np.random.normal(
                    0, feature_noise * np.sqrt(np.mean(np.square(x))), size=len(x)
                )
                for x in X_train_scaled.T
            ]
        ).T

    print("X_train:", X_train_scaled.shape)
    print("y_train:", y_train_scaled.shape)

    ##################################################
    # Run BART VIP
    ##################################################
    dataset_name = dataset.split("/")[-1][:-7]
    # Combine y and X
    dat = np.concatenate((y_train.reshape(-1, 1), X_train), axis=1)
    dat = pd.DataFrame(dat)

    # Activate the pandas to R conversion
    pandas2ri.activate()
    r = robjects.r

    # Convert the DataFrame to an R DataFrame
    r_df = pandas2ri.py2rpy(dat)

    try:
        if est_name == "BART_VIP":
            r.source("BART_VIP.R")
            runBART = robjects.globalenv["runBART"]

            # Run the R script
            t0p = time.process_time()
            t0t = time.time()
            result = runBART(r_df, random_state, rep)
            process_time = time.process_time() - t0p
            time_time = time.time() - t0t

            # Convert to list
            vip_avg = list(result[0])
            vip_rank_avg = list(result[1])
            idx_hclst = list(result[2])

        elif est_name == "BART_perm":
            r.source("BART_perm.R")
            runBART = robjects.globalenv["runBART"]

            # Run the R script
            t0p = time.process_time()
            t0t = time.time()
            result = runBART(r_df, random_state, rep)
            process_time = time.process_time() - t0p
            time_time = time.time() - t0t

            # Convert the result to list
            idx_local = list(result[0])
            idx_gmax = list(result[1])
            idx_gse = list(result[2])

        else:
            print("vs_method not recognized, exiting...")
            sys.exit(1)

    except subprocess.CalledProcessError as e:
        print(f"Error running R script: {e}")
        process_time = np.nan
        time_time = np.nan
        vip_avg = np.nan
        vip_rank_avg = np.nan
        idx_hclst = np.nan
        idx_local = np.nan
        idx_gmax = np.nan
        idx_gse = np.nan

    ##################################################
    # write to file
    ##################################################
    if est_name == "BART_VIP":
        results = {
            "dataset": dataset_name,
            "random_state": random_state,
            "process_time": process_time,
            "time_time": time_time,
            "vip": vip_avg,
            "vip_rank": vip_rank_avg,
            "idx_hclst": idx_hclst,
            "target_noise": target_noise,
            "feature_noise": feature_noise,
            "SNR": signal_to_noise,
            "n": n_samples,
            "p": features.shape[1],
            "p0": p0 if sym_data else features.shape[1],
            "rep": rep,
        }
    else:
        results = {
            "dataset": dataset_name,
            "random_state": random_state,
            "process_time": process_time,
            "time_time": time_time,
            "idx_local": idx_local,
            "idx_gmax": idx_gmax,
            "idx_gse": idx_gse,
            "target_noise": target_noise,
            "feature_noise": feature_noise,
            "SNR": signal_to_noise,
            "n": n_samples,
            "p": features.shape[1],
            "p0": p0 if sym_data else features.shape[1],
            "rep": rep,
        }
    if sym_data:
        results["true_model"] = true_model

    ##################################################
    # write to file
    ##################################################
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    save_file = (
        results_path + "/" + dataset_name + "_" + est_name + "_" + str(random_state)
    )
    if target_noise > 0:
        save_file += "_target-noise" + str(target_noise)
    if feature_noise > 0:
        save_file += "_feature-noise" + str(feature_noise)
    if signal_to_noise > 0:
        save_file += "_signal-to-noise" + str(signal_to_noise)
    if n_samples > 0:
        save_file += "_n-samples" + str(n_samples)
    if rep > 0:
        save_file += "_rep" + str(rep)
    if og_data:
        save_file += "_og_data"

    print("save_file:", save_file)

    with open(save_file + ".json", "w") as out:
        json.dump(jsonify(results), out, indent=4)

    if return_results:
        return results


################################################################################
# main entry point
################################################################################
import argparse

if __name__ == "__main__":
    import pandas as pd

    # parse command line arguments
    parser = argparse.ArgumentParser(
        description="Evaluate a method on a dataset.", add_help=False
    )
    parser.add_argument(
        "INPUT_FILE",
        type=str,
        help="Data file to analyze; ensure that the "
        'target/label column is labeled as "class".',
    )
    parser.add_argument(
        "-h", "--help", action="help", help="Show this help message and exit."
    )
    parser.add_argument(
        "-ml",
        action="store",
        dest="ALG",
        default="BART_VIP",
        type=str,
        help="Name of the variable selection method to run",
    )
    parser.add_argument(
        "-results_path",
        action="store",
        dest="RDIR",
        default="results_test",
        type=str,
        help="Name of save file",
    )
    parser.add_argument(
        "-seed",
        action="store",
        dest="RANDOM_STATE",
        default=42,
        type=int,
        help="Seed / trial",
    )
    parser.add_argument(
        "-n_jobs",
        action="store",
        type=str,
        default="1",
        help="number of cores available",
    )
    parser.add_argument(
        "-target_noise",
        action="store",
        dest="Y_NOISE",
        default=0.0,
        type=float,
        help="Gaussian noise to add" "to the target",
    )
    parser.add_argument(
        "-feature_noise",
        action="store",
        dest="X_NOISE",
        default=0.0,
        type=float,
        help="Gaussian noise to add" "to the target",
    )
    parser.add_argument(
        "-signal_to_noise",
        action="store",
        dest="SNR",
        default=0.0,
        type=float,
        help="Signal-to-noise ratio",
    )
    parser.add_argument(
        "-n",
        action="store",
        dest="N_SAMPLES",
        default=0,
        type=int,
        help="Sample size",
    )
    parser.add_argument(
        "-sym_data",
        action="store_true",
        dest="SYM_DATA",
        help="Use symbolic dataset settings",
    )
    parser.add_argument(
        "-og_data",
        action="store_true",
        dest="OG_DATA",
        default=False,
        help="For Groundtruth datasets, whether to use the \
            original dataset or add irrelevant features",
    )
    parser.add_argument(
        "-rep",
        action="store",
        dest="REP",
        default=0,
        type=int,
        help="BART replications",
    )

    args = parser.parse_args()
    set_env_vars(args.n_jobs)

    # optional keyword arguments passed to evaluate
    eval_kwargs = {}

    # check for conflicts btw cmd line args and eval_kwargs
    if args.SYM_DATA:
        eval_kwargs["scale_x"] = False
        eval_kwargs["scale_y"] = False
        eval_kwargs["sym_data"] = True

    run_BART(
        args.INPUT_FILE,
        args.RDIR,
        args.RANDOM_STATE,
        args.ALG,
        target_noise=args.Y_NOISE,
        feature_noise=args.X_NOISE,
        signal_to_noise=args.SNR,
        n_samples=args.N_SAMPLES,
        rep=args.REP,
        og_data=args.OG_DATA,
        **eval_kwargs,
    )
