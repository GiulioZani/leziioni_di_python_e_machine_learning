import pandas as pd
import numpy as np
import ipdb
from sklearn.linear_model import LinearRegression


def preprocess_data(train_data, norm_type="minmax"):
    train_data.drop(["Id"], axis=1, inplace=True)
    assert (
        not train_data[train_data.columns[-1]].isnull().any()
    ), "Ci sono NaNs nell'ultima colonna :'("
    dataframe_nan_count = len(
        [c for c in train_data.columns if train_data[c].isnull().any()]
    )
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
    return train_data.values, denorm_param1, denorm_param2


def kfold(data, k=10):
    max_div = int(k * (len(data) // k))
    fold_size = max_div // k
    data = data[:max_div]
    folds = data.reshape(k, fold_size, data.shape[-1])
    return folds


def main():
    norm_type = "norm"
    raw_data = pd.read_csv("train.csv")
    preprocessed_data, denorm_param1, denorm_param2 = preprocess_data(
        raw_data.copy(), norm_type
    )
    indici = np.random.choice(
        len(preprocessed_data), size=(len(preprocessed_data),), replace=False
    )
    preprocessed_data = preprocessed_data[indici]
    nan_count = len([c for c in preprocessed_data.T if np.isnan(c).any()])
    assert nan_count == 0, "Ci sono dei NaN!!"
    k = 10
    folds = kfold(preprocessed_data, k)
    tot_mae = []
    for i in range(k):
        val_data = folds[i]
        val_xs, val_ys = val_data[:, :-1], val_data[:, -1]
        train_data = np.concatenate((folds[:i], folds[i + 1 :]))
        train_data = np.concatenate([fold for fold in train_data])
        train_xs, train_ys = train_data[:, :-1], train_data[:, -1]
        model = LinearRegression().fit(train_xs, train_ys)
        pred_ys = model.predict(val_xs)
        if norm_type == "minmax":
            pred_ys = pred_ys * (denorm_param2 - denorm_param1) + denorm_param1
            val_ys = val_ys * (denorm_param2 - denorm_param1) + denorm_param1
        elif norm_type == "norm":
            pred_ys = pred_ys * denorm_param2 + denorm_param1
            val_ys = val_ys * denorm_param2 + denorm_param1
        mse = np.mean((pred_ys - val_ys) ** 2)
        # print(mse)
        mae = np.mean(np.abs(pred_ys - val_ys))
        print(mae)
        tot_mae.append(mae)
    mean_mae = np.mean(tot_mae)
    print(f"{raw_data.SalePrice.mean()=}")
    print(f"{mean_mae=}")


if __name__ == "__main__":
    main()
