import numpy as np
import ipdb


def knn(train_xs, train_cs, test_xs, test_cs):
    centroids = []
    for c in np.unique(train_cs):
        mask = train_cs == c
        c_xs = train_xs[mask]
        mean_c_xs = np.mean(c_xs, axis=0)
        centroids.append(mean_c_xs)
    centroids = np.array(centroids)
    # distances = np.array([(test_xs - centroid) ** 2 for centroid in centroids]).mean(
    #     axis=-1
    # )
    c = centroids.repeat(200, axis=0).reshape(200, 3, 4)
    x = test_xs.repeat(3, axis=0).reshape(3, 200, 4).transpose(1, 0, 2)
    distances = ((c - x) ** 2).mean(axis=-1).T
    predicted_labes = distances.argmin(axis=0)
    # predicted_labes = []
    # for d in distances.T:
    #     min_index = np.argmin(d)
    #     predicted_labes.append(min_index)
    # predicted_labes = np.array(predicted_labes)
    correct_predictions = predicted_labes == test_cs
    accuracy = np.sum(correct_predictions) / len(correct_predictions)
    # per ogni test_xs, calcola la distanza euclidea
    # con tutti i centroidi e sceglie il più vicino
    # questa sara' la classe predetta
    # calcola l'accuracy, in percentuale, in base al
    # numero di predizioni corrette
    return accuracy


def main():
    xs = np.random.rand(1000, 4)
    cs = np.random.randint(0, 3, (1000,))
    # suddivide i dati in train e test (80% train, 20% test)
    # (assiemen non li avevamo suddivisi)
    train_len = int(len(xs) * 0.8)
    train_xs = xs[:train_len]
    train_cs = cs[:train_len]
    test_xs = xs[train_len:]
    test_cs = cs[train_len:]

    accuracy = knn(train_xs, train_cs, test_xs, test_cs)
    print(accuracy)


if __name__ == "__main__":
    main()
