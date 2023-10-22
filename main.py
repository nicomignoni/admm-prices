#!/usr/bin/env -S python -i
import numpy as np
from agents import Retailer, Prosumer, Storage

# Number of agents
N_RETAILERS = 5
N_PROSUMERS = 3
N_STORAGES  = 2
N = N_RETAILERS + N_PROSUMERS + N_STORAGES

# Adjacency
A = np.random.randint(0, 2, (N,N))
A = (A + A.T) // 2
np.fill_diagonal(A,0)

# Time steps
T = 15

# Agents parameters
ALPHA   = 1 - 1e-4
ETA_IN  = 1 - 1e-3
ETA_OUT = 1 - 1e-3
S_MAX   = 100 * np.random.rand(N)
S0      = 
DELTA   = 20 * np.random.rand(N,T)
A_IN    = 100 * np.random.rand(N)
A_OUT   = 100 * np.random.rand(N)

MAX_X_IN  = 1e3
MAX_X_OUT = 1e3

# Iteration parameters
BETA    = 1
GAMMA_1 = 1
GAMMA_2 = 1

if __name__ == "__main__":
    retailers = [
        Retailer(
            a_in=A_IN[i], a_out=A_OUT[i], 
            N=N, T=T, is_neighbour=A[i], 
            max_x_in=MAX_X_IN, max_x_out=MAX_X_OUT, 
            beta=BETA, gamma_1=GAMMA_1
        ) \
        for i in range(N_RETAILERS)
    ]

    prosumers = [
        Prosumer(
            delta=DELTA[i],
            N=N, T=T, is_neighbour=A[i],
            max_x_in=MAX_X_IN, max_x_out=MAX_X_OUT,
            beta=BETA, gamma_1=GAMMA_1
        )
        for i in range(N_RETAILERS, N_PROSUMERS + N_RETAILERS)
    ]
