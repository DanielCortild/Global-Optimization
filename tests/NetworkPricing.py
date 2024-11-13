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
# lamify = lambda x: np.array([[x, 2], [4, x]])
lam_bar = smp.Matrix([6, 1])


p = lambda lam, beta, n, s: smp.exp(- beta * E[:, s].dot(lam[n, :])) / (smp.exp(- beta * E[:, s].dot(lam[n, :])) + smp.exp(- beta * E[:, s].dot(lam_bar)))
obj = lambda lam, beta: sum([sum([E[:, s].dot(lam[n, :]) * p(lam, beta, n, s) for s in range(S)]) for n in range(N)])

d = S * D
lam = smp.Matrix(S, D, smp.symbols('lambda:2:2'))
beta = 0.05
U_smp = lambda x: obj(x, beta)
U = lambda x: obj(smp.Matrix(np.array(x).reshape(S, D)), beta)
dU_smp = lambda x: smp.Matrix([[smp.diff(U_smp(lam), lam[i, j]).subs(list(zip(lam, x)))
    for j in range(D)] for i in range(S)])
dU = lambda x: np.array(dU_smp(smp.Matrix(np.array(x).reshape(S, D)))[:])
initial = lambda: np.random.multivariate_normal(np.zeros(d) + 3, 10 * np.eye(d))

# Create a new instance of the HRLA class
algorithm = HRLA.Algorithm(d=d, M=100, N=10, K=1400, h=0.01, title="Test", U=U, dU=dU, initial=initial)
samples_filename = algorithm.generate_samples(As=[1,2,3,4], sim_annealing=False)

# Plot empirical probabilities
postprocessor = HRLA.PostProcessor(samples_filename)
postprocessor.plot_empirical_probabilities(dpi=100, layout="32", tols=[1,2,3,4,5,6], running=False)
