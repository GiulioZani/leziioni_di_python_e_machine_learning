import numpy as np


def covar(a, b):
    """
    sum = 0
    mean_a = np.mean(a)
    mean_b = np.mean(b)
    for el_a, el_b in zip(a, b):
        sum += (el_a - mean_a)*(el_b - mean_b)
    return sum /(len(a) - 1)
    """
    return np.sum((a - np.mean(a)) * (b - np.mean(b))) / (len(a) - 1)


def main():
    a = np.array([1, 2, 3, 4, 5])
    b = np.array([2, 4, 6, 8, 10])
    print(covar(a, b))


if __name__ == "__main__":
    main()
