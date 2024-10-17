import argparse
import os

import numpy as np
import pandas as pd
import yaml
from sympy import lambdify, symbols, sympify
from tqdm import tqdm

rng = np.random.default_rng(12345)
tqdm.pandas()


def generate_dataset(
    expr_data, units, n=100000, signal_ratio=50, output_path="../feynman_dataset/"
):
    # Extract formula and variable information
    formula_str = expr_data["Formula"]
    p = expr_data["# variables"]
    x_names = [expr_data[f"v{i}_name"] for i in range(1, p + 1)]
    mins = [expr_data[f"v{i}_low"] for i in range(1, p + 1)]
    maxs = [expr_data[f"v{i}_high"] for i in range(1, p + 1)]

    # Create symbolic variables
    sympy_vars = symbols(x_names)

    # Convert the formula string to a Sympy expression
    formula_expr = sympify(formula_str, evaluate=False)

    # Generate data matrix X
    X = np.zeros((n, p))
    for i in range(p):
        X[:, i] = rng.uniform(mins[i], maxs[i], size=n)

    # Generate irrelevant variables following the same distribution
    X_bads = [rng.uniform(mins[i], maxs[i], size=(n, signal_ratio)) for i in range(p)]
    X_bad = np.hstack(X_bads)
    mins_bad = np.repeat(mins, signal_ratio)
    maxs_bad = np.repeat(maxs, signal_ratio)
    x_names_bad = np.repeat(x_names, signal_ratio)

    # Randomly permute the X_bad matrix
    idx = rng.permutation(X_bad.shape[1])
    X_bad = X_bad[:, idx]
    mins_bad = [mins_bad[i] for i in idx]
    maxs_bad = [maxs_bad[i] for i in idx]
    x_names_bad = [x_names_bad[i] for i in idx]

    # Combine X with X_bad
    X_all = np.hstack((X, X_bad))
    mins_all = mins + mins_bad
    maxs_all = maxs + maxs_bad
    x_names_all = x_names + x_names_bad

    # Create DataFrame for X with column names
    X_columns = x_names + [f"x_bad_{i+1}" for i in range(p, X_all.shape[1])]
    X_df = pd.DataFrame(X_all, columns=X_columns)

    # Create a lambda function for the formula
    formula_func = lambdify(sympy_vars, formula_expr, "numpy")

    # Evaluate formula to generate y
    y = formula_func(*[X_df[name].values for name in x_names])
    y_df = pd.DataFrame(y, columns=["target"])
    dataset = pd.concat([X_df, y_df], axis=1)

    # Save dataset to .tsv.gz
    path = os.path.normpath(output_path + "/" + expr_data["Filename"])
    # path = os.path.expanduser("~/feynman_dataset_2/") + expr_data["Filename"] + "/"
    if not os.path.exists(path):
        os.makedirs(path)
    filename = "feynman_" + expr_data["Filename"] + ".tsv.gz"
    dataset.to_csv(path + "/" + filename, sep="\t", index=False, compression="gzip")

    ### Metadata
    nl = os.linesep
    description = f"A synthetic physics model from the Feynman Lectures on Physics. This is the version of the model with units. Formula and simulated variable ranges given below.{nl}{nl}{expr_data['Output']}={expr_data['Formula']}{nl}{nl}{''.join([f'{name} in [{low},{high}]{nl}' for name, low, high in zip(X_columns, mins_all, maxs_all)])}{nl}Note the original data has been down-sampled to 100,000 rows (see source)."

    # Merge units DataFrame with features information
    x_units = pd.merge(
        units,
        pd.DataFrame(x_names_all, columns=["Variable"]),
        on="Variable",
        how="right",
    )

    # Construct feature descriptions
    x_descriptions = []
    for iii, feature in x_units.iterrows():
        tmp = feature.iloc[2:]
        nonzero_dim = tmp[tmp != 0]
        u = " * ".join(
            [
                f"{col}^{val}" if val > 0 else f"{col}^({val})"
                for col, val in nonzero_dim.items()
            ]
        )
        d = f"{feature['Units']}" if u == "" else f"{feature['Units']}, {u}"
        x_descriptions.append(d)

    metadata = {
        "dataset": expr_data["Filename"],
        "description": description,
        "source": "Feynman Symbolic Regression Database https://space.mit.edu/home/tegmark/aifeynman.html",
        "publication": "AI Feynman - a Physics-Inspired Method for Symbolic Regression, Udrescu & Tegmark (2019), arXiv:1905.11481",
        "task": "regression",
        "n": n,
        "p0": p,
        "p": X_df.shape[1],
        "signal_ratio": "1:" + str(signal_ratio),
        "keywords": ["symbolic regression", "physics"],
        "target": {"type": "continuous"},
        "features": [
            {"name": name, "type": "continuous", "description": x_description}
            for name, x_description in zip(X_columns, x_descriptions)
        ],
    }

    with open(path + "/metadata.yaml", "w") as file:
        yaml.safe_dump(metadata, file, sort_keys=False, default_flow_style=False)

    return dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate Feynman datasets with irrelevant variables.",
        add_help=False,
    )
    parser.add_argument(
        "-results",
        action="store",
        dest="RDIR",
        default="results",
        type=str,
        help="Results directory",
    )
    parser.add_argument(
        "-n",
        action="store",
        dest="N_SAMPLES",
        default=100000,
        type=int,
        help="Sample size for Feynman dataset",
    )
    parser.add_argument(
        "-signal_ratio",
        action="store",
        dest="RATIO",
        default=50,
        type=int,
        help="Number of irrelevant variables per true variable",
    )
    args = parser.parse_args()

    ##########
    # Correct yaml's behavior for storing strings with \n break
    ##########
    yaml.SafeDumper.org_represent_str = yaml.SafeDumper.represent_str  # type: ignore

    def repr_str(dumper, data):
        if "\n" in data:
            return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")
        return dumper.org_represent_str(data)

    yaml.add_representer(str, repr_str, Dumper=yaml.SafeDumper)

    ##########
    # Load Feynman formulas and units
    ##########
    units = pd.read_csv("units.csv")
    feynman = pd.read_csv("FeynmanEquations.csv")
    feynman["Filename"] = feynman["Filename"].str.replace(".", "_", regex=False)

    # Generate dataset for each Feynman formula
    feynman.progress_apply(lambda row: generate_dataset(expr_data=row, units=units, n=args.N_SAMPLE, signal_ratio=args.RATIO, output_path=args.RDIR), axis=1)  # type: ignore
