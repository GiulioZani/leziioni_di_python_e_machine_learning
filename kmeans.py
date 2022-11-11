import numpy as np
import matplotlib.pyplot as plt
import ipdb


def kmeans(data, k: int):
    # indices = np.random.randint(0, len(data), k)
    indices = np.random.choice(len(data), k, replace=False)
    centroids = data[indices]
    distances = np.array([(data - centroid) ** 2 for centroid in centroids]).mean(
        axis=-1
    )
    assigned_centroids = np.argmin(distances, axis=0)
    new_centroids = np.array([data[assigned_centroids == i] for i in range(k)]).mean()


def generate_data(size: int, k: int):
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
    plt.scatter(*data.T)
    plt.savefig("plot.png")
    final_centroids = kmeans(data, k)


if __name__ == "__main__":
    main()
