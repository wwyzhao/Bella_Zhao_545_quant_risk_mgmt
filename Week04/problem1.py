import numpy as np

n = 10000
sigma_2 = 1
np.random.seed(1000)
r_t = np.random.randn(n - 1) * np.sqrt(sigma_2)

# Use not small P_t_pre to simulate
P_t_pre = 2
print(f"Simulate with r_t ~ N(0, {sigma_2}), P_t_pre = {P_t_pre}")

# Initialize P_0 as P_t_pre
P_t_classic = np.ones(n)
P_t_arith_return = np.ones(n)
P_t_log_return = np.ones(n)
P_t_classic[0], P_t_arith_return[0], P_t_log_return[0] = P_t_pre, P_t_pre, P_t_pre


for i in range(1, n):
    # Classical Brownian Motion
    P_t_classic[i] = P_t_pre + r_t[i - 1]
    # Arithmetic Return System
    P_t_arith_return[i] = P_t_pre * (1 + r_t[i - 1])
    # Geometric Brownian Motions
    P_t_log_return[i] = P_t_pre * np.exp(r_t[i - 1])


print(f"Classical Brownian Motion: E[P] = {np.mean(P_t_classic)}, Standard Deviation[p] = {np.std(P_t_classic)}")
print(f"Arithmetic Return System: E[P] = {np.mean(P_t_arith_return)}, Standard Deviation[p] = {np.std(P_t_arith_return)}")
print(f"Geometric Brownian Motion: E[P] = {np.mean(P_t_log_return)}, Standard Deviation[p] = {np.std(P_t_log_return)}")


# Use very small P_t_pre to simulate
P_t_pre = 1e-4
print(f"Simulate with r_t ~ N(0, {sigma_2}), P_t_pre = {P_t_pre}")

# Initialize P_0 as P_t_pre
P_t_classic = np.ones(n)
P_t_arith_return = np.ones(n)
P_t_log_return = np.ones(n)
P_t_classic[0], P_t_arith_return[0], P_t_log_return[0] = P_t_pre, P_t_pre, P_t_pre

for i in range(1, n):
    # Classical Brownian Motion could give negative price
    P_t_classic[i] = P_t_pre + r_t[i - 1] 
    # Arithmetic Return System
    P_t_arith_return[i] = P_t_pre * (1 + r_t[i - 1])
    # Geometric Brownian Motions
    P_t_log_return[i] = P_t_pre * np.exp(r_t[i - 1])


print(f"Classical Brownian Motion: E[P] = {np.mean(P_t_classic)}, Standard Deviation[p] = {np.std(P_t_classic)}")
print(f"Arithmetic Return System: E[P] = {np.mean(P_t_arith_return)}, Standard Deviation[p] = {np.std(P_t_arith_return)}")
print(f"Geometric Brownian Motion: E[P] = {np.mean(P_t_log_return)}, Standard Deviation[p] = {np.std(P_t_log_return)}")
