import argparse
import json
import os

from joblib import Parallel, delayed

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Clean and evaluate symbolic accuracy.", add_help=False
    )
    parser.add_argument("INDIR", type=str, help="Input directory")
    parser.add_argument(
        "-results",
        action="store",
        dest="RDIR",
        default="results",
        type=str,
        help="Results output directory",
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
        "-n_jobs",
        action="store",
        dest="N_JOBS",
        default=int(1),
        type=int,
        help="Number of parallel jobs",
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
        help="Step in processing json files. 1 = Clean symbolic model, 2 = Simplify symbolic model, 3 = Check whether difference is a constant, 4 = Check whether ratio is a constant",
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

    # Store all commands
    all_commands = []
    n_total = 0
    n_skip = 0

    # Which key to check
    key_dict = {
        1: "symbolic_model_cleaned",
        2: "symbolic_model_simplified",
        3: "constant_diff",
        4: "constant_ratio",
        5: "complexity_simplified",
    }
    key = key_dict.get(args.STEP)

    # Traverse the input directory and gather all JSON files
    for root, _, files in os.walk(args.INDIR):
        for filename in files:
            if filename.endswith(".json"):
                n_total += 1
                input_filepath = os.path.join(root, filename)

                # Create corresponding output path
                relative_path = os.path.relpath(root, args.INDIR)
                output_filepath = os.path.join(args.RDIR, relative_path, filename)

                # Ensure the output directory exists
                os.makedirs(os.path.dirname(output_filepath), exist_ok=True)

                # Check if output file already exists, and skip if it does
                if os.path.exists(output_filepath):
                    json_data = json.load(open(output_filepath, "r"))
                    if key in json_data:
                        n_skip += 1
                        continue

                all_commands.append(
                    "python assess_symbolic_model.py "
                    "{DATASET}"
                    " -results_path {RDIR}"
                    " -data_path {DDIR}"
                    " -timeout {TIME} "
                    " -step {STEP} "
                    " -ratio {RATIO} ".format(
                        DATASET=input_filepath,
                        RDIR=output_filepath,
                        DDIR=args.DDIR,
                        TIME=args.TIME,
                        STEP=args.STEP,
                        RATIO=args.RATIO,
                    )
                )

    print(f"Found {n_skip}/{n_total} jobs with results, skipping them...")
    Parallel(n_jobs=args.N_JOBS)(
        delayed(os.system)(run_cmd) for run_cmd in all_commands
    )
    print(f"Finished Step {args.STEP}")
