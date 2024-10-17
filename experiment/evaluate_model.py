import inspect
import json
import os
import re
import shutil
import time
import warnings
from math import ceil
from unittest import result

import numpy as np
import pandas as pd
from BART_selection import run_BART
from read_file import read_file
from sklearn.base import clone
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import (
    GridSearchCV,
    HalvingGridSearchCV,
    KFold,
    train_test_split,
)
from sklearn.preprocessing import StandardScaler
from symbolic_utils import (
    add_commas,
    decompose_mrgp_model,
    find_vars_idx,
    get_sym_model,
    replace_patterns_1,
    replace_patterns_2,
    replace_patterns_3,
    zero_to_one_idx,
)
from utils import jsonify


def set_env_vars(n_jobs):
    os.environ["OMP_NUM_THREADS"] = n_jobs
    os.environ["OPENBLAS_NUM_THREADS"] = n_jobs
    os.environ["MKL_NUM_THREADS"] = n_jobs


def f1(TP: np.integer, FP: np.integer, FN: np.integer):
    if TP == 0:
        return 0
    else:
        return (2 * TP) / (2 * TP + FP + FN)


def evaluate_model(
    dataset,
    results_path,
    random_state,
    est_name,
    est,
    hyper_params,
    complexity,
    model,
    test=False,
    target_noise=0.0,
    feature_noise=0.0,
    signal_to_noise=0.0,
    # n_samples=10000,  # For blackbox datasets
    n_samples=0,  # For ground-truth datasets
    scale_x=True,
    scale_y=True,
    pre_train=None,
    skip_tuning=False,
    sym_data=False,
    og_data=False,
    vs_method=None,
    vs_result_path=None,
    vs_idx_label=None,
    rep=0,
):
    print(40 * "=", "Evaluating " + est_name + " on ", dataset, 40 * "=", sep="\n")

    np.random.seed(random_state)
    if hasattr(est, "random_state"):
        est.random_state = random_state

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
        X_test_scaled = sc_X.transform(X_test)
    else:
        X_train_scaled = X_train
        X_test_scaled = X_test

    if scale_y:
        print("scaling y")
        sc_y = StandardScaler()
        y_train_scaled = sc_y.fit_transform(y_train.reshape(-1, 1)).flatten()
        y_test_scaled = sc_y.transform(y_test.reshape(-1, 1)).flatten()
    else:
        y_train_scaled = y_train
        y_test_scaled = y_test

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

    # run any method-specific pre_train routines
    if pre_train:
        pre_train(est, X_train_scaled, y_train_scaled)

    print("X_train:", X_train_scaled.shape)
    print("y_train:", y_train_scaled.shape)

    ##################################################
    # Load BART result if vs_method is not None
    ##################################################
    if vs_method is not None:
        print(f"Setting prescreening to {vs_method} and using {vs_idx_label}")
        dataset_name = dataset.split("/")[-1][:-7]

        # Load pre-run results
        if vs_result_path is not None:
            feather_path = os.path.normpath(vs_result_path)
            print("Loading prescreening result from " + feather_path)
            BART_result = pd.read_feather(feather_path)

            # Locate corresponding col_idx
            idx = (
                (BART_result["dataset_name"] == dataset_name)
                & (BART_result["random_state"] == random_state)
                & (BART_result["SNR"] == signal_to_noise)
                & (BART_result["n"] == n_samples)
            )

            col_idx = BART_result[vs_idx_label][idx].values[0].astype(int)

        # Run BART in-place
        else:
            n_str = "" if n_samples == 0 else f"n_{n_samples}"
            result_dir = "results_feynman" if sym_data else "results_blackbox"
            BART_result = run_BART(
                dataset=dataset,
                results_path=os.path.join(
                    os.path.dirname(__file__),
                    f"../{result_dir}/{vs_method}/{n_str}/{dataset_name}",
                ),
                random_state=random_state,
                est_name=vs_method,
                target_noise=target_noise,
                feature_noise=feature_noise,
                signal_to_noise=signal_to_noise,
                n_samples=n_samples,
                rep=rep,
                scale_x=scale_x,
                scale_y=scale_y,
                sym_data=sym_data,
                og_data=og_data,
                return_results=True,
            )
            col_idx = BART_result.get(vs_idx_label)

        # If we didn't select any column, copy old result and exit
        if np.array_equal(col_idx, np.array([])):
            SNR_str = (
                ""
                if signal_to_noise == 0.0
                else "_signal-to-noise" + str(signal_to_noise)
            )
            if sym_data:
                source_path = os.path.normpath(
                    "../results_feynman/SR/n_"
                    + str(n_samples)
                    + "/"
                    + dataset_name
                    + "/"
                    + dataset_name
                    + "_"
                    + est_name
                    + "_"
                    + str(random_state)
                    + SNR_str
                    + "_n-samples"
                    + str(n_samples)
                    + ".json"
                )
            else:
                source_path = os.path.normpath(
                    +"../results_blackbox/SR/"
                    + dataset_name
                    + "/"
                    + dataset_name
                    + "_"
                    + est_name
                    + "_"
                    + str(random_state)
                    + ".json"
                )

            # Copy the file
            if os.path.exists(source_path):
                shutil.copy2(source_path, results_path)
                print("Did not select any column. Copying old results. Script ending.")
            else:
                print("Source path not found.")

            # End the script
            exit()

        # Select useful columns
        print("Pre-screening with", vs_method)
        print("BART selected columns:", col_idx)
        X_train_scaled = X_train_scaled[:, col_idx]
        X_test_scaled = X_test_scaled[:, col_idx]
        feature_names = feature_names[col_idx]
        print("New feature names:", feature_names)
        print("X_train_BART:", X_train_scaled.shape)
        print("y_train:", y_train_scaled.shape)

    ##################################################
    # define CV strategy for hyperparam tuning
    ##################################################
    # define a test mode with fewer splits, no hyper_params, and few iterations
    if test:
        print("test mode enabled")
        n_splits = 2
        hyper_params = {}
        print("hyper_params set to", hyper_params)
        for genname in [
            "generations",
            "gens",
            "g",
            "itrNum",
            "treeNum",
            "evaluations",
            "niterations",
        ]:
            if hasattr(est, genname):
                print("setting", genname, "=2 for test")
                setattr(est, genname, 2)
        for popname in ["popsize", "pop_size", "population_size", "val", "npop"]:
            if hasattr(est, popname):
                print("setting", popname, "=20 for test")
                setattr(est, popname, 20)
        if hasattr(est, "BF_try_time"):
            setattr(est, "BF_try_time", 1)
        if hasattr(est, "NN_epochs"):
            setattr(est, "NN_epochs", 1)
        for timename in ["time", "max_time", "time_out", "time_limit"]:
            if hasattr(est, timename):
                print("setting", timename, "= 10 for test")
                setattr(est, timename, 10)
        # deep sr setting
        if hasattr(est, "config"):
            est.config["training"]["n_samples"] = 10
            est.config["training"]["batch_size"] = 10
            est.config["training"]["hof"] = 5
    else:
        n_splits = 5

    if skip_tuning:
        print("skipping tuning")
        grid_est = est if "uDSR" in est_name else clone(est)
    else:
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

        grid_est = HalvingGridSearchCV(
            est,
            cv=cv,
            param_grid=hyper_params,
            verbose=2,
            n_jobs=1,
            scoring="r2",
            error_score=0.0,
        )

    ##################################################
    # Fit models
    ##################################################
    print("training", grid_est)
    print("Parameters: ", grid_est.get_params())
    t0p = time.process_time()
    t0t = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        grid_est.fit(X_train_scaled, y_train_scaled)

    process_time = time.process_time() - t0p
    time_time = time.time() - t0t
    print("Training time measures:", time_time)
    best_est = grid_est if skip_tuning else grid_est.best_estimator_
    # best_est = grid_est

    ##################################################
    # store results
    ##################################################
    dataset_name = dataset.split("/")[-1][:-7]
    if vs_method is not None:
        est_name = est_name + "." + vs_method
    results = {
        "dataset": dataset_name,
        "algorithm": est_name,
        "params": jsonify(best_est.get_params()),
        "random_state": random_state,
        "process_time": process_time,
        "time_time": time_time,
        "target_noise": target_noise,
        "feature_noise": feature_noise,
        "SNR": signal_to_noise,
        "n": n_samples,
        "p": features.shape[1],
        "p0": p0 if sym_data else features.shape[1],
        "vs_method": vs_method,
    }
    if sym_data:
        results["true_model"] = true_model

    # get the size of the final model
    if complexity == None:
        results["model_size"] = int(features.shape[1])
    else:
        results["model_size"] = int(complexity(best_est))

    # get the final symbolic model as a string
    if model == None:
        results["symbolic_model"] = "not implemented"
    else:
        if "X" in inspect.signature(model).parameters.keys():
            results["symbolic_model"] = model(best_est, X_train_scaled)
        else:
            results["symbolic_model"] = model(best_est)

        # find TP, FP, TN, FN, F1
        if sym_data:
            model_str = results["symbolic_model"]
            mrgp = "MRGP" in est_name

            if mrgp:
                model_str = model_str.replace("+", "add")
                model_str = add_commas(model_str)
                betas, model_str = decompose_mrgp_model(model_str)

            # Change to consistent feature names: X_number
            model_str = replace_patterns_1(model_str)
            model_str = replace_patterns_2(model_str)
            model_str = replace_patterns_3(model_str)

            # Rename X_number to the actural feature name
            if any([n in est_name.lower() for n in ["mrgp", "operon", "dsr"]]):
                new_model_str = model_str
            else:
                new_model_str = zero_to_one_idx(model_str)  # Change 0-index to 1-index

            # Find selected variables and their indice
            selected_idx = find_vars_idx(new_model_str)  # 1-index
            idx = np.array(selected_idx) - 1  # 0-index

            # Calculate selection metrics
            selected_name = list(feature_names[idx]) if idx.size > 0 else []
            n_features = len(selected_idx)
            incorrect_name = re.compile(r"\bx_bad_\d+")
            FP = sum(1 for name in selected_name if incorrect_name.search(name))
            TP = n_features - FP
            FN = p0 - TP
            TN = p - TP - FP - FN
            F1 = f1(TP, FP, FN)

            results["TP"] = TP
            results["FP"] = FP
            results["TN"] = TN
            results["FN"] = FN
            results["F1"] = F1
            results["selected_name"] = selected_name
            results["selected_idx"] = selected_idx
            results["perf_metric_updated"] = True

    # scores
    pred = grid_est.predict

    results["failed"] = "no error"
    for fold, target, X in zip(
        ["train", "test"],
        [y_train_scaled, y_test_scaled],
        [X_train_scaled, X_test_scaled],
    ):
        # y_pred = sc_y.inverse_transform(pred(X)) if scale_y else pred(X)
        y_pred = np.asarray(pred(X)).reshape(-1, 1).flatten()
        if scale_y:
            y_pred = sc_y.inverse_transform(y_pred.reshape(-1, 1)).flatten()
            target = sc_y.inverse_transform(target.reshape(-1, 1)).flatten()

        for score, scorer in [
            ("mse", mean_squared_error),
            ("mae", mean_absolute_error),
            ("r2", r2_score),
        ]:
            try:
                results[score + "_" + fold] = scorer(target, y_pred)
            except Exception as e:
                results[score + "_" + fold] = np.nan
                results["failed"] = "NaN in prediction"
                print(
                    f"Exception occurred: {e}. Assigned np.nan to results[{score + '_' + fold}]"
                )
    results["version"] = "v2"

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
    if vs_method is not None:
        save_file += "_rep" + str(rep)
    if og_data:
        save_file += "_og_data"

    print("save_file:", save_file)

    with open(save_file + ".json", "w") as out:
        json.dump(jsonify(results), out, indent=4)


################################################################################
# main entry point
################################################################################
import argparse
import importlib

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
        default=None,
        type=str,
        help="Name of estimator (with matching file in methods/)",
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
        "-test",
        action="store_true",
        dest="TEST",
        help="Used for testing a minimal version",
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
        "-skip_tuning",
        action="store_true",
        dest="SKIP_TUNE",
        default=False,
        help="Dont tune the estimator",
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
        "-vs_method",
        action="store",
        dest="VS_METHOD",
        default=None,
        type=str,
        help="Variable selection method for prescreening",
    )
    parser.add_argument(
        "-vs_result_path",
        action="store",
        dest="VS_PATH",
        default=None,
        type=str,
        help="Directory of the prescreening result",
    )
    parser.add_argument(
        "-vs_idx_label",
        action="store",
        dest="VS_IDX",
        default=None,
        type=str,
        help="Label of the prescreening result in feather or JSON.",
    )
    parser.add_argument(
        "-rep",
        action="store",
        dest="REP",
        default=0,
        type=int,
        help="Number of BART replications",
    )

    args = parser.parse_args()
    set_env_vars(args.n_jobs)
    print("import from", "methods." + args.ALG)
    algorithm = importlib.__import__("methods." + args.ALG, globals(), locals(), ["*"])

    print("algorithm:", algorithm.est)
    if "hyper_params" not in dir(algorithm):
        algorithm.hyper_params = {}
    print("hyperparams:", algorithm.hyper_params)

    ##################################################
    # Load tuned parameters
    ##################################################
    if not args.SYM_DATA:
        blackbox = pd.read_feather(
            os.path.normpath("../results_srbench/black-box_results.feather")
        )

        # Locate tuned parameters based on
        # 1. dataset
        # 2. random_state
        # 3. algorithm
        cond_dataset = blackbox["dataset"] == args.INPUT_FILE.split("/")[-1][:-7]
        cond_random_state = blackbox["random_state"] == args.RANDOM_STATE

        # Some names in feather are different
        algo_name = args.ALG.split(".")[-1]
        algo_name = algo_name.replace("Regressor", "")
        if algo_name == "FE_AFP":
            algo_name = "AFP_FE"
        elif algo_name == "sembackpropgp":
            algo_name = "SBP-GP"
        elif algo_name == "GPGOMEA":
            algo_name = "GP-GOMEA"

        cond_algorithm = blackbox["algorithm"] == algo_name

        # Find their intersection
        combined_condition = cond_dataset & cond_random_state & cond_algorithm

        # If intersection is not False, apply tuned parameters
        if combined_condition.any():
            pp = blackbox.loc[combined_condition, "params_str"]
            params = eval(pp.iloc[0])

            # Reset MAXTIME
            MAXTIME = int(24 * 60 * 60)
            if "max_time" in dir(algorithm.est):
                params["max_time"] = MAXTIME
            elif "time_limit" in dir(algorithm.est):
                params["time_limit"] = MAXTIME
            elif "time" in dir(algorithm.est):
                params["time"] = MAXTIME
            elif "time_out" in dir(algorithm.est):
                params["time_out"] = MAXTIME

            # Reset 'tmp_dir' to None for AIFeynman
            if algo_name == "AIFeynman":
                params["tmp_dir"] = None

            if algo_name == "gplearn":
                if "const_range" in params:
                    params["const_range"] = eval(params["const_range"])
                if "function_set" in params:
                    params["function_set"] = eval(params["function_set"])
                if "init_depth" in params:
                    params["init_depth"] = eval(params["init_depth"])

            if algo_name == "Operon":
                if "btc_bias" in params:
                    params.pop("btc_bias")
                if "error_metric" in params:
                    # Rename the key
                    params["objectives"] = [params.pop("error_metric")]

            # Set tuned parameters
            algorithm.est.set_params(**params)

        if algo_name == "PySR":
            algorithm.est.random_state = args.RANDOM_STATE

    # optional keyword arguments passed to evaluate
    eval_kwargs = {}
    if "eval_kwargs" in dir(algorithm):
        eval_kwargs = algorithm.eval_kwargs

    # check for conflicts btw cmd line args and eval_kwargs
    if args.SYM_DATA:
        eval_kwargs["scale_x"] = False
        eval_kwargs["scale_y"] = False
        eval_kwargs["skip_tuning"] = True
        eval_kwargs["sym_data"] = True
    if args.SKIP_TUNE:
        eval_kwargs["skip_tuning"] = True

    evaluate_model(
        args.INPUT_FILE,
        args.RDIR,
        args.RANDOM_STATE,
        args.ALG,
        algorithm.est,
        algorithm.hyper_params,
        algorithm.complexity,
        algorithm.model,
        test=args.TEST,
        target_noise=args.Y_NOISE,
        feature_noise=args.X_NOISE,
        signal_to_noise=args.SNR,
        n_samples=args.N_SAMPLES,
        og_data=args.OG_DATA,
        vs_method=args.VS_METHOD,
        vs_result_path=args.VS_PATH,
        vs_idx_label=args.VS_IDX,
        rep=args.REP,
        **eval_kwargs,
    )
