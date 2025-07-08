import numpy as np
import GlobalOptimizationHRLA as GO
from PostProcessing import Comparator

# Define Rastrigin function, its gradient and an initial distribution
d = 10
M = 100
N = 10
K = 14000
h = 0.01

U = lambda x: d + np.linalg.norm(x) ** 2 - np.sum(np.cos(2*np.pi*x))
dU = lambda x: 2 * x + 2 * np.pi * np.sin(2*np.pi*x)
initial = lambda: np.random.multivariate_normal(np.zeros(d) + 3, 10 * np.eye(d))

# Compute iterates according to HRLA
titleHRLA = f"HRLA"
algorithmHRLA = GO.HRLA(d=d, M=M, N=N, K=K, h=h, title=titleHRLA, U=U, dU=dU, initial=initial)
samples_filename_HRLA = algorithmHRLA.generate_samples(As=[2,3,4,5], sim_annealing=False)

# Compute iterates according to ULA
titleULA = f"ULA"
algorithmULA = GO.ULA(d=d, M=M, N=N, K=K, h=h, title=titleULA, U=U, dU=dU, initial=initial)
samples_filename_ULA = algorithmULA.generate_samples(As=[2,3,4,5], sim_annealing=False)

# Compute iterates according to OLA
titleOLA = f"OLA"
algorithmOLA = GO.OLA(d=d, M=M, N=N, K=K, h=h, title=titleOLA, U=U, dU=dU, initial=initial)
samples_filename_OLA = algorithmOLA.generate_samples(As=[2,3,4,5], sim_annealing=False)

# Compare results
comparator = Comparator([samples_filename_ULA, samples_filename_OLA, samples_filename_HRLA])
comparator.plot_empirical_probabilities_per_a_e(dpi=10, tols=[2, 4], As=[4, 3], running=False)

