import numpy as np
from scipy.stats import norm
data = np.array([1.2, 2.3, 0.7, 1.6, 1.1, 1.8, 0.9, 2.2])

# Initialize parameters
mu1, mu2 = 0.0, 1.0
sigma1, sigma2 = 1.0, 1.0
p1, p2 = 0.5, 0.5

# EM Algorithm
for iteration in range(10):
    # E-step
    likelihood1 = norm.pdf(data, mu1, sigma1)
    likelihood2 = norm.pdf(data, mu2, sigma2)

    weight1 = p1 * likelihood1 / (p1 * likelihood1 + p2 * likelihood2)
    weight2 = p2 * likelihood2 / (p1 * likelihood1 + p2 * likelihood2)

    # M-step
    mu1 = np.sum(weight1 * data) / np.sum(weight1)
    mu2 = np.sum(weight2 * data) / np.sum(weight2)

    sigma1 = np.sqrt(np.sum(weight1 * (data - mu1) ** 2) / np.sum(weight1))
    sigma2 = np.sqrt(np.sum(weight2 * (data - mu2) ** 2) / np.sum(weight2))

    p1 = np.mean(weight1)
    p2 = np.mean(weight2)

# Final parameters
print("Final Parameters after EM:")
print("Mean 1:", mu1)
print("Mean 2:", mu2)
print("Sigma 1:", sigma1)
print("Sigma 2:", sigma2)
print("Mixing Coefficient p1:", p1)
print("Mixing Coefficient p2:", p2)
