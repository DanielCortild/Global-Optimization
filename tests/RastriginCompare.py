import numpy as np
import GlobalOptimizationHRLA as GO

# Define Rastrigin function, its gradient and an initial distribution
d = 10
U = lambda x: d + np.linalg.norm(x) ** 2 - np.sum(np.cos(2*np.pi*x))
dU = lambda x: 2 * x + 2 * np.pi * np.sin(2*np.pi*x)
initial = lambda: np.random.multivariate_normal(np.zeros(d) + 3, 10 * np.eye(d))

# Compute iterates according to HRLA
titleHRLA = "Rastrigin HRLA"
algorithmHRLA = GO.HRLA(d=d, M=100, N=10, K=14000, h=0.01, title=titleHRLA, U=U, dU=dU, initial=initial)
samples_filename_HRLA = algorithmHRLA.generate_samples(As=[1,2,3,4], sim_annealing=False)
postprocessorHRLA = GO.PostProcessor(samples_filename_HRLA)
postprocessorHRLA.plot_empirical_probabilities(dpi=100, layout="22", tols=[1,2,3,4], running=False)

# Compute iterates according to ULA
titleULA = "Rastrigin ULA"
algorithmULA = GO.ULA(d=d, M=100, N=10, K=14000, h=0.01, title=titleULA, U=U, dU=dU, initial=initial)
samples_filename_ULA = algorithmULA.generate_samples(As=[1,2,3,4], sim_annealing=False)
postprocessorULA = GO.PostProcessor(samples_filename_ULA)
postprocessorULA.plot_empirical_probabilities(dpi=100, layout="22", tols=[1,2,3,4], running=False)

# Compute iterates according to OLA
titleOLA = "Rastrigin OLA"
algorithmOLA = GO.OLA(d=d, M=100, N=10, K=14000, h=0.01, title=titleOLA, U=U, dU=dU, initial=initial)
samples_filename_OLA = algorithmOLA.generate_samples(As=[1,2,3,4], sim_annealing=False)
postprocessorOLA = GO.PostProcessor(samples_filename_OLA)
postprocessorOLA.plot_empirical_probabilities(dpi=100, layout="22", tols=[1,2,3,4], running=False)
