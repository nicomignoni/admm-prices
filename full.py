#!/usr/bin/env -S python -i

import numpy as np
import cvxpy as cp

N = 4
T = 5

# Create the graph and make it undirected
A = np.random.randint(0, 2, (N,N))
A = (A + A.T) // 2
