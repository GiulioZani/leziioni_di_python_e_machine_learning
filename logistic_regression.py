import pandas as pd
import numpy as np
import ipdb


def main():
    df = pd.read_csv("weatherAUS.csv")
    df.drop(["Date", "Location"], axis=1, inplace=True)
    # df.dropna(inplace=True)
    nan_percentage = [(n, c.isnull().sum() / df.shape[0]) for n, c in df.items()]
    filterd_nan_percentage = [n for n, p in nan_percentage if p < 0.1]
    print(df.shape)
    print(len(filterd_nan_percentage))
    print(filterd_nan_percentage)
    df = df[filterd_nan_percentage]
    df.fillna(method="ffill", inplace=True)
    print(f"Dataframe contains nans: {df.isnull().any().any()}")
    norm_type = "minmax"  # o norm
    norm_matrix = np.zeros(df.T.shape)
    for i, (_, column) in enumerate(df.items()):
        if column.dtype in (int, float):
            if norm_type == "minmax":
                norm_matrix[i] = (
                    (column - column.min()) / (column.max() - column.min())
                ).values
            elif norm_type == "norm":
                norm_matrix[i] = ((column - column.mean()) / column.std()).values
            else:
                raise ValueError("Norm type not recognized")
        else:
            raw_values = column.values.tolist()
            possible_values = list(set(raw_values))
            indices = np.array([possible_values.index(value) for value in raw_values])
            if norm_type == "minmax":
                # if column_name == "WindGustDir":
                #     ipdb.set_trace()
                norm_indices = (indices - indices.min()) / (
                    indices.max() - indices.min()
                )
                norm_matrix[i] = norm_indices

            elif norm_type == "norm":
                norm_indices = (indices - indices.mean()) / indices.std()
                norm_matrix[i] = norm_indices


if __name__ == "__main__":
    main()
