import numpy as np
import GlobalOptimizationHRLA as GO

# Define Rastrigin function, its gradient and an initial distribution
d = 10
U = lambda x: d + np.linalg.norm(x) ** 2 - np.sum(np.cos(2*np.pi*x))
dU = lambda x: 2 * x + 2 * np.pi * np.sin(2*np.pi*x)
initial = lambda: np.random.multivariate_normal(np.zeros(d) + 3, 10 * np.eye(d))

# Compute iterates according to HRLA
# combs = [[5, 1, 1], [5, 1, 2], [5, 2, 1], [5, 2, 2]]
# for i, (b, alpha, beta) in enumerate(combs):
print()
# print(f"HRLA with b={b}, alpha={alpha}, beta={beta}")
titleHRLA = f"HRLA"
algorithmHRLA = GO.HRLA(d=d, M=100, N=10, K=14000, h=0.01, title=titleHRLA, U=U, dU=dU, initial=initial)
samples_filename_HRLA = algorithmHRLA.generate_samples(As=[2,3,4,5], sim_annealing=False)
postprocessorHRLA = GO.PostProcessor(samples_filename_HRLA)
postprocessorHRLA.plot_empirical_probabilities(dpi=100, layout="22", tols=[1,2,3,4], running=False)

# Compute iterates according to ULA
# combs = [[10, 1], [10, 0.1], [10, 100], [10, 10], [1, 1], [1, 10], [1, 0.1], [100, 1], [100, 10], [100, 0.1]]
# for i, (b, alpha) in enumerate(combs):
#     print()
#     print(f"ULA with b={b}, alpha={alpha}")
titleULA = f"ULA"
algorithmULA = GO.ULA(d=d, M=100, N=10, K=14000, h=0.01, title=titleULA, U=U, dU=dU, initial=initial)
samples_filename_ULA = algorithmULA.generate_samples(As=[2,3,4,5], sim_annealing=False)
postprocessorULA = GO.PostProcessor(samples_filename_ULA)
postprocessorULA.plot_empirical_probabilities(dpi=100, layout="22", tols=[1,2,3,4], running=False)

titleOLA = f"OLA"
algorithmOLA = GO.OLA(d=d, M=100, N=10, K=14000, h=0.01, title=titleOLA, U=U, dU=dU, initial=initial)
samples_filename_OLA = algorithmOLA.generate_samples(As=[2,3,4,5], sim_annealing=False)
postprocessorOLA = GO.PostProcessor(samples_filename_OLA)
postprocessorOLA.plot_empirical_probabilities(dpi=100, layout="22", tols=[1,2,3,4], running=False)

# Compare results
comparator = GO.Comparator([samples_filename_HRLA, samples_filename_ULA, samples_filename_OLA])
comparator.plot_empirical_probabilities_per_a(dpi=100, tol=5, running=False, layout="22")
comparator.plot_empirical_probabilities_per_a(dpi=100, tol=4, running=False, layout="22")
comparator.plot_empirical_probabilities_per_a(dpi=100, tol=3, running=False, layout="22")
comparator.plot_empirical_probabilities_per_a(dpi=100, tol=2, running=False, layout="22")
comparator.plot_empirical_probabilities_per_a(dpi=100, tol=1, running=False, layout="22")


# Compute iterates according to OLA
# combs = [0.1, 0.5, 1, 2, 10]
# for i, alpha in enumerate(combs):
#     print()
#     print(f"OLA with alpha={alpha}")
#     titleOLA = f"Rastrigin OLA{i}"
#     algorithmOLA = GO.OLA(d=d, M=100, N=10, K=14000, h=0.01, title=titleOLA, U=U, dU=dU, initial=initial, alpha=alpha)
#     samples_filename_OLA = algorithmOLA.generate_samples(As=[1,2,3,4], sim_annealing=False)
#     postprocessorOLA = GO.PostProcessor(samples_filename_OLA)
#     postprocessorOLA.plot_empirical_probabilities(dpi=100, layout="22", tols=[1,2,3,4], running=False)

# titleOLA = "Rastrigin OLA2"
# algorithmOLA = GO.OLA(d=d, M=100, N=10, K=14000, h=0.01, title=titleOLA, U=U, dU=dU, initial=initial, alpha=10)
# samples_filename_OLA = algorithmOLA.generate_samples(As=[1,2,3,4], sim_annealing=False)
# postprocessorOLA = GO.PostProcessor(samples_filename_OLA)
# postprocessorOLA.plot_empirical_probabilities(dpi=100, layout="22", tols=[1,2,3,4], running=False)

# titleOLA = "Rastrigin OLA3"
# algorithmOLA = GO.OLA(d=d, M=100, N=10, K=14000, h=0.01, title=titleOLA, U=U, dU=dU, initial=initial, alpha=0.1)
# samples_filename_OLA = algorithmOLA.generate_samples(As=[1,2,3,4], sim_annealing=False)
# postprocessorOLA = GO.PostProcessor(samples_filename_OLA)
# postprocessorOLA.plot_empirical_probabilities(dpi=100, layout="22", tols=[1,2,3,4], running=False)

