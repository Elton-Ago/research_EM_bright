import numpy as np
import matplotlib.pyplot as plt
import bilby
import matplotlib.pyplot as plt
from bilby.core.likelihood import GaussianLikelihood
from bilby.core.prior import Uniform
import seaborn

np.random.seed(123)
seaborn.set_context("talk")
seaborn.set_style("whitegrid")

def modelA(time, omega):
    return np.sin(omega*time)

sigma = 0.5
omega_true = 1.2
time = np.linspace(0, 12, 100)
ydet = modelA(time, omega=omega_true)
yobs = ydet + np.random.normal(0, sigma, len(time))

plt.plot(time, yobs, "o")
plt.ylabel("$y_{obs}$"); plt.xlabel('time')
plt.show()

def ln_likelihood(yobs, time, omega, sigma=0.1):
    yA = modelA(time, omega)
    ln_likes = -0.5 * ((yobs - yA)**2 / sigma**2 + np.log(2*np.pi*sigma*2))
    return np.sum(ln_likes)

omega_grid = np.linspace(omega_true-2e-2, omega_true+2e-2, 1000)
ln_likelihood_grid = []
for omega in omega_grid:
    ln_likelihood_grid.append(ln_likelihood(yobs, time, omega))
    
plt.plot(omega_grid, ln_likelihood_grid)
plt.axvline(omega_true, ls='--')
plt.xlabel("$\omega$")
plt.ylabel("Log-likelihood")
plt.show()

max_likelihood_omega = omega_grid[np.argmax(ln_likelihood_grid)]
plt.plot(time, yobs, "-x")
plt.plot(time, modelA(time, max_likelihood_omega))
plt.show()

pdf_unnormalized = np.exp(ln_likelihood_grid - np.mean(ln_likelihood_grid))
pdf_normalized = pdf_unnormalized / np.sum(pdf_unnormalized)
plt.plot(omega_grid, pdf_normalized)
plt.axvline(omega_true, ls='--')
plt.xlabel("$\omega$")
plt.ylabel("PDF")
plt.show()

cdf = np.cumsum(pdf_normalized)
plt.plot(omega_grid, cdf)
plt.xlabel("$\omega$")
plt.ylabel("CDF")
plt.show()

median = omega_grid[np.argmin(np.abs(cdf - 0.5))]
low_bound = omega_grid[np.argmin(np.abs(cdf - 0.05))]
upper_bound = omega_grid[np.argmin(np.abs(cdf - 0.95))]

plt.plot(omega_grid, pdf_normalized)
plt.axvline(omega_true, ls='--')
plt.fill_between([low_bound, upper_bound], 0, 2 * np.max(pdf_normalized), color = "C2", alpha=0.2)
plt.axvline(median, color="C2")
plt.ylim(0, 1.1 * np.max(pdf_normalized)); plt.xlabel("$\omega$"); plt.ylabel("PDF")
plt.show()

omega_values = [1.16] #pick a starting point
ln_likelihood_values = [ln_likelihood(yobs, time, omega_values[0])]

for i in range(100):
    proposed_point = omega_values[-1] + np.random.normal(0, 0.01)
    ln_likelihood_proposed = ln_likelihood(yobs, time, proposed_point)
    
    if ln_likelihood_proposed > ln_likelihood_values[-1]:
        omega_values.append(proposed_point)
        ln_likelihood_values.append(ln_likelihood_proposed)
        
plt.plot(omega_values, "-x")
plt.show()

#MCMC samplers use randomness to sample from the posterior distribution

omega_values = [1.18] #pick a starting point
ln_likelihood_values = [ln_likelihood(yobs, time, omega_values[0])]

for i in range(100):
    proposed_point = omega_values[-1] + np.random.normal(0, 0.01)
    ln_likelihood_proposed = ln_likelihood(yobs, time, proposed_point)
    
    if ln_likelihood_proposed > ln_likelihood_values[-1] + np.log(np.random.rand()):
        omega_values.append(proposed_point)
        ln_likelihood_values.append(ln_likelihood_proposed)
    else:
        omega_values.append(omega_values[-1])
        ln_likelihood_values.append(ln_likelihood_values[-1])
    
plt.plot(omega_values, '-x')
plt.show()

omega_values = [1.18] #pick from starting point
ln_likelihood_values = [ln_likelihood(yobs, time, omega_values[0])]

for i in range(10000):
    proposed_point = omega_values[-1] + np.random.normal(0, 0.01)
    ln_likelihood_proposed = ln_likelihood(yobs, time, proposed_point)
    
    if ln_likelihood_proposed > ln_likelihood_values[-1] + np.log(np.random.rand()):
        omega_values.append(proposed_point)
        ln_likelihood_values.append(ln_likelihood_proposed)
    else:
        omega_values.append(omega_values[-1])
        ln_likelihood_values.append(ln_likelihood_values[-1])
        
plt.plot(omega_values, "-x")
plt.show()

histogram = plt.hist(omega_values, bins=50)

#stochastic samplers are all about drawing "samples" from the posterior distribution
#simple MCMC methods need tuning
#advanced MCMC methods have fewer tuning parameters
#nested sampling is better than MCMC

#Now using Bilby

likelihood = bilby.likelihood.GaussianLikelihood(time, yobs, modelA, sigma=0.1)

ln_likelihood_grid_bilby = []
for omega in omega_grid:
    likelihood.parameters["omega"] = omega
    ln_likelihood_grid_bilby.append(likelihood.log_likelihood())
    
plt.plot(omega_grid, np.exp(ln_likelihood_grid - np.mean(ln_likelihood_grid)), label="No Bilby")
plt.plot(omega_grid, np.exp(ln_likelihood_grid_bilby - np.mean(ln_likelihood_grid_bilby)), '--', label='Bilby')
plt.legend()
plt.show()

#what happens if our model looks like equation of motion for a wave y(t) = A*sin(omega*t + phi0) and we don't know
#what sigma is? we can create a 4D grid and evaluate the likelihood over the grid and find a maximum

#stochastic sampling methods as a blackbox

def modelB(time, omega, A, phi0):
    return A*np.sin(omega*time + phi0)

likelihood = bilby.likelihood.GaussianLikelihood(time, yobs, modelB)

priors = dict(
    A = Uniform(0, 2, "A"),
    omega = Uniform(1, 1.5, "omega"),
    phi0 = Uniform(-np.pi, np.pi, "phi0", boundary='reflective'),
    sigma = Uniform(0, 2, "sigma"))

result = bilby.run_sampler(
    likelihood, priors=priors, nlive=500, sample='unif', outdir="test-outdir", label='black-box',
    injection_parameters=dict(A=1, omega=omega_true, phi0 = 0, sigma=sigma),
    clean=True)

result.plot_corner()
