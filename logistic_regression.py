import matplotlib.pyplot as plt
import numpy as np

# Define the logistic function
def logistic_function(x, u, s):
    return 1 / (1 + np.exp(-(x - u) / s))


def main():
    # Define the x-axis
    x = np.linspace(-10, 10, 1000)

    # Define the parameters
    a = 1
    b = 0

    # Compute the logistic function
    y = logistic_function(x, a, b)

    # Plot the results
    plt.plot(x, y)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Logistic function")
    plt.show()


if __name__ == "__main__":
    main()
