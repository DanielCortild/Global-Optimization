import numpy as np
import GlobalOptimizationHRLA as HRLA

# Define Rastrigin function, its gradient and an initial distribution
d = 10
U = lambda x: d + np.linalg.norm(x) ** 2 - np.sum(np.cos(2*np.pi*x))
dU = lambda x: 2 * x + 2 * np.pi * np.sin(2*np.pi*x)
initial = lambda: np.random.multivariate_normal(np.zeros(d) + 3, 10 * np.eye(d))

# Compute iterates according to algorithm
title = "Rastrigin HRLA"
algorithm = HRLA.AlgorithmHRLA(d=d, M=100, N=10, K=14000, h=0.01, title=title, U=U, dU=dU, initial=initial)
samples_filename_HRLA = algorithm.generate_samples(As=[1,2,3,4], sim_annealing=False)

# Compute iterates according to ULA
title = "Rastrigin ULA"
algorithm = HRLA.AlgorithmULA(d=d, M=100, N=10, K=14000, h=0.01, title=title, U=U, dU=dU, initial=initial)
samples_filename_ULA = algorithm.generate_samples(As=[1,2,3,4], sim_annealing=False)

# Plot empirical probabilities
postprocessor = HRLA.PostProcessor(samples_filename_HRLA)
postprocessor.plot_empirical_probabilities(dpi=1, layout="22", tols=[1,2,3,4], running=False)

postprocessor = HRLA.PostProcessor(samples_filename_ULA)
postprocessor.plot_empirical_probabilities(dpi=1, layout="22", tols=[1,2,3,4], running=False)

# Compute table of averages and standard deviations
postprocessor.compute_tables([5, 14], 1, "mean")
postprocessor.compute_tables([5, 14], 1, "std")
