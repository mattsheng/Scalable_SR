import argparse
import json
import multiprocessing
import os

import numpy as np
import sympy as sp
from symbolic_utils import clean_pred_model, symplify_model
from utils import jsonify


def save(r, output_filepath):
    with open(output_filepath, "w") as f:
        json.dump(jsonify(r), f, indent=4)


def run_step(input_file, output_file, step, ratio, data_path):
    if step == 1:
        clean_symbolic_model(input_file, output_file, data_path)
    elif step == 2:
        simplify_symbolic_model(input_file, output_file, ratio, data_path)
    elif step == 3:
        check_diff_constant(input_file, output_file, data_path)
    elif step == 4:
        check_ratio_constant(input_file, output_file, data_path)
    else:
        update_comlexity(input_file, output_file, data_path)


def complexity(sp_model):
    c = 0
    for _ in sp.preorder_traversal(sp_model):
        c += 1
    return c


# 1) Clean `symbolic_model`
def clean_symbolic_model(input_filepath, output_filepath, data_path="feynman"):
    # Read the JSON file
    if os.path.exists(input_filepath):
        json_data = json.load(open(input_filepath, "r"))
    else:
        raise FileNotFoundError(input_filepath + " not found")

    # Process the JSON data
    try:
        sym_model = json_data["symbolic_model"]
        if sym_model.strip() != "":
            dataset_name = json_data["dataset"].replace("feynman_", "")
            metadata_path = os.path.join(data_path, dataset_name, "metadata.yaml")
            est_name = json_data["algorithm"]
            sym_model_cleaned = clean_pred_model(
                model_str=sym_model,
                metadata=metadata_path,
                est_name=est_name,
            )
            json_data["symbolic_model_cleaned"] = str(sym_model_cleaned)

        else:
            json_data["symbolic_model_cleaned"] = ""

    except Exception as e:
        print(f"Error processing Step 1 (clean) for {input_filepath}: {e}")
        json_data["sympy_exception_step1"] = str(e)
        if "symbolic_model_cleaned" not in json_data.keys():
            json_data["symbolic_model_cleaned"] = "clean_failed"
            json_data["error_clean"] = True

    # Write the processed result to the output file
    save(json_data, output_filepath)


# 2) Simplify `symbolic_model_cleaned`
def simplify_symbolic_model(
    input_filepath, output_filepath, ratio=1, data_path="feynman"
):
    # Read the JSON file
    if os.path.exists(input_filepath):
        json_data = json.load(open(input_filepath, "r"))
    else:
        raise FileNotFoundError(input_filepath + " not found")

    # Check if Step 1 failed or handle edge cases
    if "symbolic_model_cleaned" not in json_data.keys():
        print(f"`symbolic_model_cleaned` not found in {input_filepath}")
        json_data["symbolic_model_simplified"] = "simplify_failed"
    elif json_data["symbolic_model_cleaned"] == "clean_failed":
        print(f"clean_failed in Step 1. Skipping Step 2 for {input_filepath}")
        json_data["symbolic_model_simplified"] = "simplify_failed"
    elif json_data["symbolic_model_cleaned"] == "":
        json_data["symbolic_model_simplified"] = ""
    else:
        try:
            model_str = json_data["symbolic_model_cleaned"]
            dataset_name = json_data["dataset"].replace("feynman_", "")
            metadata_path = os.path.join(data_path, dataset_name, "metadata.yaml")

            # Symplify model string first
            sym_model_cleaned = symplify_model(model_str, metadata_path)

            # Simplify `sym_model_cleaned`
            sym_model_simplified = sp.simplify(sym_model_cleaned, ratio=ratio)
            json_data["symbolic_model_simplified"] = str(sym_model_simplified)

        except Exception as e:
            print(f"Error processing Step 2 (simplify) for {input_filepath}: {e}")
            json_data["sympy_exception_step2"] = str(e)
            json_data["error_simplify"] = True
            if "symbolic_model_simplified" not in json_data.keys():
                json_data["symbolic_model_simplified"] = json_data.get(
                    "symbolic_model_cleaned", "simplify_failed"
                )

    # Write the processed result to the output file
    save(json_data, output_filepath)


# 3) Check diff is constant
def check_diff_constant(input_filepath, output_filepath, data_path="feynman"):
    # Read the JSON file
    if os.path.exists(input_filepath):
        json_data = json.load(open(input_filepath, "r"))
    else:
        raise FileNotFoundError(input_filepath + " not found")

    bad_model_cleaned = ["clean_failed", "nan", "", "0", "", "zoo"]
    bad_model_simplified = ["simplify_failed", "nan", "", "0", "", "zoo"]

    # If clean failed, then simplify must also be failed
    if json_data["symbolic_model_cleaned"] not in bad_model_cleaned:
        # Only check constant error for decent model
        if json_data["r2_test"] > 0.5:
            # If simplify failed, but clean didn't, use cleaned model
            if json_data["symbolic_model_simplified"] in bad_model_simplified:
                model_str = json_data["symbolic_model_cleaned"]
            # If simplify succeed, use simplified model
            else:
                model_str = json_data["symbolic_model_simplified"]

            # Checking whether diff is constant
            try:
                dataset_name = json_data["dataset"].replace("feynman_", "")
                metadata_path = os.path.join(data_path, dataset_name, "metadata.yaml")

                # Symplify model string first
                sym_model_simplified = symplify_model(model_str, metadata_path)
                true_model = symplify_model(json_data["true_model"], metadata_path)

                # Simplify `sym_model_cleaned`
                diff = sym_model_simplified - true_model
                result = diff.is_constant()
                json_data["constant_diff"] = False if result is None else result

            except Exception as e:
                print(f"Error processing Step 3 (check diff) for {input_filepath}: {e}")
                json_data["sympy_exception_step3"] = str(e)
                json_data["error_diff"] = True
                if "constant_diff" not in json_data.keys():
                    json_data["constant_diff"] = False
        else:
            json_data["constant_diff"] = False
    else:
        json_data["constant_diff"] = False
        json_data["constant_ratio"] = False

    # Write the processed result to the output file
    save(json_data, output_filepath)


# 4) Check ratio is constant
def check_ratio_constant(input_filepath, output_filepath, data_path="feynman"):
    # Read the JSON file
    if os.path.exists(input_filepath):
        json_data = json.load(open(input_filepath, "r"))
    else:
        raise FileNotFoundError(input_filepath + " not found")

    bad_model_simplified = ["simplify_failed", "nan", "", "0", "", "zoo"]

    # Check if `constant_ratio` has been set to False in Step 3
    if "constant_ratio" not in json_data.keys() and json_data["r2_test"] > 0.5:
        # If simplify failed, but clean didn't, use cleaned model
        # Don't need to check cleaned model anymore
        if json_data["symbolic_model_simplified"] in bad_model_simplified:
            model_str = json_data["symbolic_model_cleaned"]

        # If simplify succeed, use simplified model
        else:
            model_str = json_data["symbolic_model_simplified"]

        # Checking whether ratio is constant
        try:
            dataset_name = json_data["dataset"].replace("feynman_", "")
            metadata_path = os.path.join(data_path, dataset_name, "metadata.yaml")

            # Symplify model string first
            sym_model_simplified = symplify_model(model_str, metadata_path)
            true_model = symplify_model(json_data["true_model"], metadata_path)

            # Simplify `sym_model_cleaned`
            ratio = sym_model_simplified / true_model
            result = ratio.is_constant()
            json_data["constant_ratio"] = False if result is None else result

        except Exception as e:
            print(f"Error processing Step 4 (check ratio) for {input_filepath}: {e}")
            json_data["sympy_exception_step4"] = str(e)
            json_data["error_ratio"] = True
            if "constant_ratio" not in json_data.keys():
                json_data["constant_ratio"] = False
    else:
        json_data["constant_ratio"] = False

    # Write the processed result to the output file
    save(json_data, output_filepath)


# 5) Update complexity
def update_comlexity(input_filepath, output_filepath, data_path="feynman"):
    # Read the JSON file
    if os.path.exists(input_filepath):
        json_data = json.load(open(input_filepath, "r"))
    else:
        raise FileNotFoundError(input_filepath + " not found")

    dataset_name = json_data["dataset"].replace("feynman_", "")
    metadata_path = os.path.join(data_path, dataset_name, "metadata.yaml")

    bad_model_cleaned = ["clean_failed", "nan", "", "0", "", "zoo"]
    bad_model_simplified = ["simplify_failed", "nan", "", "0", "", "zoo"]

    if json_data["symbolic_model_simplified"] not in bad_model_simplified:
        model_str = json_data["symbolic_model_simplified"]

        try:
            sym_model = symplify_model(model_str, metadata_path, eval=False)
            json_data["complexity_simplified"] = complexity(sym_model)
        except Exception as e:
            print(f"Error processing Step 5 (complexity) for {input_filepath}: {e}")
            json_data["complexity_simplified"] = np.nan
            json_data["error_complexity"] = True

    elif json_data["symbolic_model_cleaned"] not in bad_model_cleaned:
        model_str = json_data["symbolic_model_cleaned"]

        try:
            sym_model = symplify_model(model_str, metadata_path, eval=False)
            json_data["complexity_simplified"] = complexity(sym_model)
        except Exception as e:
            print(f"Error processing Step 5 (complexity) for {input_filepath}: {e}")
            json_data["complexity_simplified"] = np.nan
            json_data["error_complexity"] = True

    else:
        json_data["complexity_simplified"] = np.nan

    # Write the processed result to the output file
    save(json_data, output_filepath)


def handle_timeout(input_filepath, output_filepath, step=1):
    """
    This function is called when a timeout occurs. It writes default values to the JSON file.
    """
    # Read the JSON file
    if os.path.exists(input_filepath):
        json_data = json.load(open(input_filepath, "r"))
    else:
        raise FileNotFoundError(input_filepath + " not found")

    # Set default values when timeout occurs
    if step == 1:
        json_data["symbolic_model_cleaned"] = "clean_timeout"
        json_data["timeout_clean"] = True
    elif step == 2:
        json_data["symbolic_model_simplified"] = json_data.get(
            "symbolic_model_cleaned", "simplify_timeout"
        )
        json_data["timeout_simplify"] = True
    elif step == 3:
        json_data["constant_diff"] = False
        json_data["timeout_diff"] = True
    elif step == 4:
        json_data["constant_ratio"] = False
        json_data["timeout_ratio"] = True
    else:
        json_data["complexity_simplified"] = np.nan
        json_data["timeout_complexity"] = True

    # Write the processed result to the output file
    save(json_data, output_filepath)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a method on a dataset.", add_help=False
    )
    parser.add_argument("INPUT_FILE", type=str, help="JSON file to analyze")
    parser.add_argument(
        "-results_path",
        action="store",
        dest="OUTPUT_FILE",
        default="results",
        type=str,
        help="Name of output file",
    )
    parser.add_argument(
        "-data_path",
        action="store",
        dest="DDIR",
        default="",
        type=str,
        help="Directory of the dataset path",
    )
    parser.add_argument(
        "-timeout",
        action="store",
        dest="TIME",
        default=int(5 * 60),
        type=int,
        help="Time limit in seconds",
    )
    parser.add_argument(
        "-step",
        action="store",
        dest="STEP",
        default=int(1),
        type=int,
        help="Step in processing json files. 1 = Clean symbolic model, 2 = Simplify symbolic model, 3 = Check whether difference is a constant, 4 = Check whether ratio is a constant, 5 = Update model complexity",
    )
    parser.add_argument(
        "-ratio",
        action="store",
        dest="RATIO",
        default=1.0,
        type=float,
        help="Ratio for sympy simplification",
    )

    args = parser.parse_args()

    p = multiprocessing.Process(
        target=run_step,
        args=(args.INPUT_FILE, args.OUTPUT_FILE, args.STEP, args.RATIO, args.DDIR),
    )
    p.start()
    p.join(args.TIME)  # Wait for the process to finish or timeout

    if p.is_alive():
        print(
            f"Step {args.STEP}: Timeout reached for {args.INPUT_FILE}, terminating the process."
        )
        p.terminate()
        p.join()
        handle_timeout(args.INPUT_FILE, args.OUTPUT_FILE, args.STEP)
