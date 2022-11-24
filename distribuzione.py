import numpy as np
import matplotlib.pyplot as plt
#from scipy.signal import gaussian


def gaussian(x, mu, sigma):
    norm_p = 1 / (sigma * np.sqrt(2 * np.pi))
    return norm_p * np.exp(-1 / 2 * ((x - mu) / sigma) ** 2)


def main():
    # normal distribution
    mu, sigma = 1.838, 0.71**4
    xs = np.linspace(mu - 3, mu + 3, 50)
    probs = gaussian(xs, mu, sigma)
    values = np.random.normal(mu, sigma, 1000)
    #plt.plot(xs, values, "-")
    plt.hist(values, bins=50, density=True)
    plt.plot(xs, probs, "-")
    plt.savefig("distribuzione.png")


if __name__ == "__main__":
    main()
