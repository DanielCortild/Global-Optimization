import numpy as np
import GlobalOptimizationHRLA as HRLA

# Define Rastrigin function, its gradient and an initial distribution
d = 15
title = "Rastrigin"
U = lambda x: d + np.linalg.norm(x) ** 2 - np.sum(np.cos(2*np.pi*x))
dU = lambda x: 2 * x + 2 * np.pi * np.sin(2*np.pi*x)
initial = lambda: np.random.multivariate_normal(np.zeros(d) + 3, 10 * np.eye(d))

# Compute iterates according to algorithm
algorithm = HRLA.AlgorithmHRLA(d=d, M=100, N=10, K=100000, h=0.01, title=title, U=U, dU=dU, initial=initial)
samples_filename = algorithm.generate_samples(As=[4,5,6,7,8], sim_annealing=False)

# Plot empirical probabilities
postprocessor = HRLA.PostProcessor(samples_filename)
postprocessor.plot_empirical_probabilities(dpi=10, layout="32", tols=[3,4,5,6,7,8], running=False)
