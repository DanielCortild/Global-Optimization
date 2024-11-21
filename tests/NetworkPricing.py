import sys
sys.path.append("../src")
import GlobalOptimizationHRLA as HRLA
import numpy as np
import sympy as smp

# Define the function to be optimized
E = smp.Matrix([
    [10, 120],
    [130, 10]
])
S = 2
N = 2
D = 2
lam_bar = smp.Matrix([6, 1])
beta = 0.05

print(f"Starting Network Pricing problem with S={S}, N={N}, D={D}, beta={beta}, E={E}, lam_bar={lam_bar}")

p = lambda lam, beta, n, s: smp.exp(- beta * E[:, s].dot(lam[n, :])) / (smp.exp(- beta * E[:, s].dot(lam[n, :])) + smp.exp(- beta * E[:, s].dot(lam_bar)))
obj = lambda lam, beta: sum([sum([E[:, s].dot(lam[n, :]) * p(lam, beta, n, s) for s in range(S)]) for n in range(N)])

d = S * D
lam = smp.Matrix(S, D, smp.symbols('lambda:2:2'))
U_smp = lambda x: -obj(x, beta)
U = lambda x: -obj(smp.Matrix(np.array(x).reshape(S, D)), beta)
diffs = [[smp.diff(U_smp(lam), lam[i, j]) for j in range(D)] for i in range(S)]
dU_smp = lambda x: smp.Matrix([[diffs[i][j].subs(list(zip(lam, x)))
    for j in range(D)] for i in range(S)])
dU = lambda x: np.array(dU_smp(smp.Matrix(np.array(x).reshape(S, D)))[:])
initial = lambda: np.random.multivariate_normal(np.zeros(d) + 3, 10 * np.eye(d))

# Create a new instance of the HRLA class
algorithm = HRLA.Algorithm(d=d, M=100, N=10, K=3000, h=0.01, title="NetworkPricing", U=U, dU=dU, initial=initial)
samples_filename = algorithm.generate_samples(As=[1,2,3,4], sim_annealing=False)
# samples_filename = "temp_output/data/NetworkPricing_1731671966.125818.pickle"

# Plot empirical probabilities
postprocessor = HRLA.PostProcessor(samples_filename)
postprocessor.compute_tables([1000, 2000, 3000], 100, "mean")
postprocessor.compute_tables([1000, 2000, 3000], 200, "best")
# postprocessor.compute_tables([700, 1400], 10, "std")
postprocessor.get_best([1000, 2000, 3000], 100)
postprocessor.plot_curves(100)