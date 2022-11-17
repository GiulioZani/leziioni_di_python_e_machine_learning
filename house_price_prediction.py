import pandas as pd
import numpy as np
import ipdb
from sklearn.linear_model import LinearRegression

def main():
    train_data = pd.read_csv("train.csv")
    train_data.drop(["Id"], axis=1, inplace=True)
    norm_type = "minmax"  # oppure "norm"
    assert (
        not train_data[train_data.columns[-1]].isnull().any()
    ), "Ci sono NaNs nell'ultima colonna :'("
    dataframe_nan_count = len(
        [c for c in train_data.columns if train_data[c].isnull().any()]
    )
    n_colonne = len(train_data.columns)
    print(f"Originalmente nel dataset c'erano {dataframe_nan_count} colonne con NaNs")
    train_data.dropna(axis=1, inplace=True)
    denorm_param1 = 0
    denorm_param2 = 0
    for column_name in train_data.columns:
        column = train_data[column_name]
        if column.dtype in (int, float):
            if norm_type == "minmax":
                train_data[column_name] = (column - column.min()) / (
                    column.max() - column.min()
                )
                if column_name == train_data.columns[-1]:
                    denorm_param1 = column.min()
                    denorm_param2 = column.max()
            elif norm_type == "norm":
                train_data[column_name] = (column - column.mean()) / column.std()
                if column_name == train_data.columns[-1]:
                    denorm_param1 = column.mean()
                    denorm_param2 = column.std()
            else:
                raise ValueError("Norm type not recognized")
        elif column.dtype == object:
            raw_values = column.values.tolist()
            possible_values = list(set(raw_values))
            indices = np.array([possible_values.index(value) for value in raw_values])
            norm_indices = (indices - indices.min()) / (indices.max() - indices.min())
            train_data[column_name] = pd.Series(norm_indices)
    preprocessed_data = train_data.values
    indici = np.random.choice(len(train_data), size=(len(train_data),), replace=False)
    shuffled_data = preprocessed_data[indici]
    nan_count = len([c for c in shuffled_data.T if np.isnan(c).any()])
    assert nan_count == 0, "Ci sono dei NaN!!"
    assert shuffled_data.shape[1] == (n_colonne - dataframe_nan_count)
    train_size = int(len(shuffled_data)*0.8)
    train_data = shuffled_data[:train_size]#
    val_data = shuffled_data[train_size:]
    train_xs = train_data[:,:-1]
    train_ys = train_data[:, -1]
    val_xs = val_data[:,:-1]
    val_ys = val_data[:, -1]
    model = LinearRegression().fit(train_xs, train_ys)
    pred_y = model.predict(val_xs)
    denorm_pred_ys = pred_ys # qualcosa
    denorm_val_ys = val_ys # qualcosa
    mse = np.mean((pred_y - val_ys)**2)
    print(mse)

if __name__ == "__main__":
    main()
