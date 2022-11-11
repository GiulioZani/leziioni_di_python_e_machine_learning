import numpy as np


def main():
    a = np.random.multivariate_normal([0, 0], [[0.2, 0], [0, 0.22]], 1000)
    b = np.random.multivariate_normal([2, 2], [[0.2, 0], [0, 0.22]], 1000)
    c = np.random.multivariate_normal([-2, -2], [[0.2, 0], [0, 0.22]], 1000)
    xs = np.concatenate((a, b, c), axis=0)
    labels = np.concatenate((np.zeros(1000), np.ones(1000), np.ones(1000)  + 1))
    # plot the results
    import matplotlib.pyplot as plt
    plt.plot(a[:, 0], a[:, 1], 'x')
    plt.plot(b[:, 0], b[:, 1], 'x')
    plt.plot(c[:, 0], c[:, 1], 'x')
    plt.axis('equal')
    plt.savefig("multinormal.png")

if __name__ == "__main__":
    main()
