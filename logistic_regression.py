import pandas as pd
import numpy as np
from tqdm import tqdm

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


def evaulate_models(models: list):
    df = pd.read_csv("weatherAUS.csv")
    df.drop(["Date"], axis=1, inplace=True)
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

    denorm_param1 = 0
    denorm_param2 = 0
    for i, (column_name, column) in enumerate(df.items()):
        print(f"Normalizing {column_name}")
        if column.dtype in (int, float):
            if norm_type == "minmax":
                norm_matrix[i] = (
                    (column - column.min()) / (column.max() - column.min())
                ).values
                if column_name == df.columns[-1]:
                    denorm_param1 = column.min()
                    denorm_param2 = column.max()
            elif norm_type == "norm":
                norm_matrix[i] = ((column - column.mean()) / column.std()).values
                if column_name == df.columns[-1]:
                    denorm_param1 = column.mean()
                    denorm_param2 = column.std()
            else:
                raise ValueError("Norm type not recognized")
        else:
            raw_values = column.values.tolist()
            possible_values = list(set(raw_values))
            indices = np.array([possible_values.index(value) for value in raw_values])
            if norm_type == "minmax":
                norm_indices = (indices - indices.min()) / (
                    indices.max() - indices.min()
                )
                norm_matrix[i] = norm_indices

            elif norm_type == "norm":
                norm_indices = (indices - indices.mean()) / indices.std()
                norm_matrix[i] = norm_indices
    k = 10
    folds = kfold(norm_matrix.T, k)

    total_accuracies = []
    for i in tqdm(range(k)):
        val_data = folds[i]
        val_xs, val_ys = val_data[:, :-1], val_data[:, -1]
        train_data = np.concatenate((folds[:i], folds[i + 1 :]))
        train_data = np.concatenate([fold for fold in train_data])
        train_xs, train_ys = train_data[:, :-1], train_data[:, -1]
        accuracies = []
        total_accuracies.append(accuracies)
        for model in models:
            trained_model = model.fit(train_xs, train_ys)
            pred_ys_l1 = trained_model.predict(val_xs)
            acc = accuracy_score(pred_ys_l1, val_ys)
            accuracies.append(acc)

    accuracy_media = np.mean(total_accuracies, axis=0)
    print(accuracy_media)


def kfold(data, k=10):
    max_div = int(k * (len(data) // k))
    fold_size = max_div // k
    data = data[:max_div]
    folds = data.reshape(k, fold_size, data.shape[-1])
    return folds


class Model:
    def __init__(self):
        pass

    def predict(self, x):
        return [1] * len(x)

    def fit(self, xs, ys):
        return self


def main():
    # models = [LogisticRegression(max_iter=1000), SVC()]
    models = [Model(), Model()]
    evaulate_models(models)


if __name__ == "__main__":
    main()
