import numpy as np
import sys
import GlobalOptimizationHRLA as GO

# Define Rastrigin function, its gradient and an initial distribution
M = 100
N = 10
K = 14000
h = 0.01

U10 = lambda x: 10 + np.linalg.norm(x) ** 2 - np.sum(np.cos(2*np.pi*x))
dU10 = lambda x: 2 * x + 2 * np.pi * np.sin(2*np.pi*x)
initial10 = lambda: np.random.multivariate_normal(np.zeros(10) + 3, 10 * np.eye(10))

# Compute iterates according to HRLA
titleHRLA10 = f"HRLA10"
algorithmHRLA10 = GO.HRLA(d=10, M=M, N=N, K=K, h=h, title=titleHRLA10, U=U10, dU=dU10, initial=initial10)
samples_filename_HRLA10 = algorithmHRLA10.generate_samples(As=[1,2,3,4], sim_annealing=False)

U20 = lambda x: 20 + np.linalg.norm(x) ** 2 - np.sum(np.cos(2*np.pi*x))
dU20 = lambda x: 2 * x + 2 * np.pi * np.sin(2*np.pi*x)
initial20 = lambda: np.random.multivariate_normal(np.zeros(20) + 3, 10 * np.eye(20))

titleHRLA20 = f"HRLA20"
algorithmHRLA20 = GO.HRLA(d=20, M=M, N=N, K=K, h=h, title=titleHRLA20, U=U20, dU=dU20, initial=initial20)
samples_filename_HRLA20 = algorithmHRLA20.generate_samples(As=[1,2,3,4], sim_annealing=False)

# Compare results
comparator = GO.Comparator([samples_filename_HRLA10, samples_filename_HRLA20])
comparator.plot_empirical_probabilities_per_d_e(dpi=10, tols=[2, 4], ds=[10, 20], running=False)
