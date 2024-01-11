total_outcomes = 6
favorable_outcomes = 1 
probability_4 = favorable_outcomes / total_outcomes
print(f"Probability of rolling a 4: {probability_4}")
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, poisson, binom, expon
mean = 50
std_dev = 10
samples = np.random.normal(mean, std_dev, 1000)
plt.figure(figsize=(8, 6))
plt.hist(samples, bins=30, density=True, alpha=0.6, color='blue')
x = np.linspace(mean - 4*std_dev, mean + 4*std_dev, 100)
plt.plot(x, norm.pdf(x, mean, std_dev), 'r-', lw=2, label='Normal Distribution')
plt.title('Normal Distribution Example (Quality Control)')
plt.xlabel('Values')
plt.ylabel('Probability Density')
plt.legend()
plt.grid(True)
plt.show()
lambda_param = 5 
k = 3 
prob_3_events = poisson.pmf(k, lambda_param)
print(f"Probability of 3 events occurring in an hour: {prob_3_events}")
n = 10 
p = 0.6 
k_success = 7 
prob_7_success = binom.pmf(k_success, n, p)
print(f"Probability of 7 successes out of 10 trials: {prob_7_success}")
exp_samples = np.random.exponential(scale=2, size=1000)
plt.figure(figsize=(8, 6))
plt.hist(exp_samples, bins=30, density=True, alpha=0.6, color='green')
x_exp = np.linspace(0, 10, 100)
plt.plot(x_exp, expon.pdf(x_exp, scale=2), 'r-', lw=2, label='Exponential Distribution')
plt.title('Exponential Distribution Example (Reliability Analysis)')
plt.xlabel('Values')
plt.ylabel('Probability Density')
plt.legend()
plt.grid(True)
plt.show()
