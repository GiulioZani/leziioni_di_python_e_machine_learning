import pandas as pd
import numpy as np
import ipdb

def main():
    train_data = pd.read_csv("train.csv")
    train_data.drop(["Id"], axis=1, inplace=True)
    norm_type = "minmax" # oppure "norm"
    for column_name in train_data.columns:
        column = train_data[column_name]
        if column.dtype in (int, float):
            train_data[column_name] = (column - column.min()) / (
                column.max() - column.min()
            )
        elif column.dtype == object:
            raw_values = column.values.tolist()
            possible_values = list(set(raw_values))
            indices = np.array([possible_values.index(value) for value in raw_values])
            norm_indices = (indices - indices.min()) / (indices.max() - indices.min())
            train_data[column_name] = pd.Series(norm_indices)

    matrice = train_data.values



if __name__ == "__main__":
    main()
