import numpy as np
import pandas as pd
import yaml


def read_file(filename, label="target", sep=None):
    if filename.endswith("gz"):
        compression = "gzip"
    else:
        compression = None

    print("compression:", compression)
    print("filename:", filename)

    if sep:
        input_data = pd.read_csv(filename, sep=sep, compression=compression)
    else:
        input_data = pd.read_csv(
            filename, sep=sep, compression=compression, engine="python"
        )

    feature_names = [x for x in input_data.columns.values if x != label]
    feature_names = np.array(feature_names)

    X = input_data.drop(label, axis=1).values.astype(float)
    y = input_data[label].values

    assert X.shape[1] == feature_names.shape[0]

    return X, y, feature_names


def read_features(filename):
    with open(filename, "r") as f:
        data = yaml.safe_load(f)

    feature_names = [feature["name"] for feature in data["features"]]

    return feature_names
