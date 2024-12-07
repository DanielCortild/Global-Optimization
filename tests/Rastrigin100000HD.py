import numpy as np
import GlobalOptimizationHRLA as GO

# Define Rastrigin function, its gradient and an initial distribution
d = 10
U = lambda x: d + np.linalg.norm(x) ** 2 - np.sum(np.cos(2*np.pi*x))
dU = lambda x: 2 * x + 2 * np.pi * np.sin(2*np.pi*x)
initial = lambda: np.random.multivariate_normal(np.zeros(d) + 3, 10 * np.eye(d))

# Compute iterates according to algorithm
title10 = "HRLA10"
algorithm10 = GO.HRLA(d=d, M=100, N=10, K=100000, h=0.01, title=title10, U=U, dU=dU, initial=initial)
samples_filename10 = algorithm10.generate_samples(As=[4,5,6,7,8], sim_annealing=False)

# Plot empirical probabilities
postprocessor10 = GO.PostProcessor(samples_filename10)
postprocessor.plot_empirical_probabilities(dpi=10, layout="32", tols=[3,4,5,6,7,8], running=False)

# Define Rastrigin function, its gradient and an initial distribution
d = 20
U = lambda x: d + np.linalg.norm(x) ** 2 - np.sum(np.cos(2*np.pi*x))
dU = lambda x: 2 * x + 2 * np.pi * np.sin(2*np.pi*x)
initial = lambda: np.random.multivariate_normal(np.zeros(d) + 3, 10 * np.eye(d))

# Compute iterates according to algorithm
title20 = "HRLA20"
algorithm20 = GO.HRLA(d=d, M=100, N=10, K=100000, h=0.01, title=title20, U=U, dU=dU, initial=initial)
samples_filename20 = algorithm20.generate_samples(As=[4,5,6,7,8], sim_annealing=False)

# Create comparator
comparator = GO.Comparator([samples_filename10, samples_filename20])
comparator.plot_empirical_probabilities_per_d(dpi=10, tols=[2,4], running=False)
