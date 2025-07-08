import numpy as np
import GlobalOptimizationHRLA as GO
from PostProcessing import PostProcessor, Comparator

# Define Rastrigin function, its gradient and an initial distribution
d = 2
U = lambda x: d + np.linalg.norm(x) ** 2 - np.sum(np.cos(2*np.pi*x))
dU = lambda x: 2 * x + 2 * np.pi * np.sin(2*np.pi*x)
initial = lambda: np.random.multivariate_normal(np.zeros(d) + 3, 10 * np.eye(d))

# Compute iterates according to algorithm
title = "HRLA"
algorithm = GO.HRLA(d=d, M=1, N=1, K=14, h=0.01, title=title, U=U, dU=dU, initial=initial)
samples_filename_HRLA = algorithm.generate_samples(As=[1,2,3,4], sim_annealing=False)

# Compute iterates according to ULA
title = "ULA"
algorithm = GO.ULA(d=d, M=1, N=1, K=14, h=0.01, title=title, U=U, dU=dU, initial=initial)
samples_filename_ULA = algorithm.generate_samples(As=[1,2,3,4], sim_annealing=True)

# Plot empirical probabilities
postprocessor = PostProcessor(samples_filename_HRLA)
postprocessor.plot_empirical_probabilities(dpi=1, layout="22", tols=[1,2,3,4], running=False)

# Compute table of averages and standard deviations
postprocessor.compute_tables([5, 14], 1, "mean")
postprocessor.compute_tables([5, 14], 1, "std")

# Create comparator
comparator = Comparator([samples_filename_HRLA, samples_filename_ULA])
comparator.plot_empirical_probabilities_per_d(dpi=1, tols=[3,4], running=True)