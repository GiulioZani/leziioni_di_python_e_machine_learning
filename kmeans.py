import numpy as np
import matplotlib.pyplot as plt
import ipdb


def kmeans(data, k: int):
    # indices = np.random.randint(0, len(data), k)
    indices = np.random.choice(len(data), k, replace=False)
    centroids = data[indices]
    centroids_distance = 1
    i = 0
    while centroids_distance > 0.0:
        print(f"{i=}")
        distances = np.array([(data - centroid) ** 2 for centroid in centroids]).mean(
            axis=-1
        )
        assigned_centroids = np.argmin(distances, axis=0)
        new_centroids = np.array(
            [data[assigned_centroids == i].mean(axis=0) for i in range(k)]
        )
        centroids_distance = np.array((new_centroids - centroids) ** 2).mean()
        centroids = new_centroids
        i += 1

    return centroids


def generate_data(size: int, k: int):  # genera dati di lunghezza size*k
    return np.concatenate(
        [
            np.random.multivariate_normal(
                [10 * np.random.rand(), 10 * np.random.rand()],
                [[0.2, 0], [0, 0.2]],
                size,
            )
            for _ in range(k)
        ]
    )


def main():
    k = 4
    data = generate_data(1000, k)
    centroids = kmeans(data, k)
    plt.scatter(*data.T)
    plt.scatter(*centroids.T)
    plt.savefig("plot.png")


if __name__ == "__main__":
    main()
