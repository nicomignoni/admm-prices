#!/usr/bin/env -S python -i
from datetime import datetime

import numpy as np
import cvxpy as cp
from agents import Retailer, Prosumer, Storage

import matplotlib.pyplot as plt

np.random.seed(2023)

RESULTS_DIR = "result"
PLOTS_DIR   = "plot"

# Number of agents
N_RETAILERS = 1
N_PROSUMERS = 1
N_STORAGES  = 1
N = N_RETAILERS + N_PROSUMERS + N_STORAGES

# Adjacency
A = np.random.randint(0, 2, (N,N))
A = (A + A.T) // 2
np.fill_diagonal(A,0)

# Time steps
T = 2

# Agents parameters
ALPHA   = 1 - 1e-4
ETA_IN  = 1 - 1e-3
ETA_OUT = 1 - 1e-3
S_MAX   = np.random.randint(50, 100, N)
S0      = np.random.rand(N,N)
S0      /= S0.sum(axis=1, keepdims=True)
S0      *= 0
DELTA   = 20 * np.random.rand(N,T)
A_IN    = 1e3 * np.random.rand(N)
A_OUT   = 1e3 * np.random.rand(N)

MAX_X_IN  = 1e3
MAX_X_OUT = 1e3

# Iteration parameters
K_ACTUAL = 1000
K_PLOT   = 150
BETA     = 0.5
GAMMA_1  = 1e-2
GAMMA_2  = 30
RHO      = 1

if __name__ == "__main__":

    # Price function
    g = lambda x: x.sum(axis=0) + 0.05

    # Intialize agents
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

    storages = [
        Storage(
            s0=S0[i], s_max=S_MAX[i], alpha=ALPHA, eta_in=ETA_IN, eta_out=ETA_OUT,
            N=N, T=T, is_neighbour=A[i],
            max_x_in=MAX_X_IN, max_x_out=MAX_X_OUT,
            beta=BETA, gamma_1=GAMMA_1
        )
        for i in range(N_RETAILERS + N_PROSUMERS, N)
    ]


    # Initialize variables and parameters
    x_in  = np.zeros((N,N,T,K+1))
    x_out = np.zeros((N,N,T,K+1))
    u_in  = np.zeros((N,N,T,K+1)) 
    u_out = np.zeros((N,N,T,K+1))
    p     = np.zeros((N,N,T,K+1))
    y     = np.zeros((N,N,T,K+1))
    l     = np.zeros((N,N,T,K+1))
    # Bx_in  = np.zeros((N,N,T,K+1))
    # Bx_out = np.zeros((N,N,T,K+1))

    # Iterative algorithm
    for k in range(K_ACTUAL):
        print(f"Iteration {k}: ")
        for i,agent in enumerate(retailers + prosumers + storages):
            # Assign the parameters to agents
            agent.old_x_in.value  = x_in[i,:,:,k]
            agent.old_x_out.value = x_out[i,:,:,k]
            agent.u_in.value      = u_in[i,:,:,k]
            agent.u_out.value     = u_out[i,:,:,k]
            agent.p.value         = p[i,:,:,k]
            agent.p_ext.value     = p[:,i,:,k]
            agent.y.value         = y[i,:,:,k]
            agent.y_ext.value     = y[:,i,:,k]

            agent.prob.solve()
            # print(f"Agent {i}, status: {agent.prob.status}")

            # Add the solutions at iteration k
            x_in[i,:,:,k+1]  = agent.x_in.value
            x_out[i,:,:,k+1] = agent.x_out.value

            # Bx_in[:,i,:,k+1]  = agent.x_in.value
            # Bx_out[:,i,:,k+1] = agent.x_out.value


            # Update the parameters
            l[i,:,:,k+1] = (BETA*(x_in[i,:,:,k+1] + x_out[:,i,:,k+1]) + GAMMA_2*y[:,i,:,k] + u_in[i,:,:,k] + u_out[:,i,:,k]) / (GAMMA_2 + 2*BETA)
            y[i,:,:,k+1] = (BETA*(x_in[:,i,:,k+1] + x_out[i,:,:,k+1]) + GAMMA_2*y[i,:,:,k] + u_in[:,i,:,k] + u_out[i,:,:,k]) / (GAMMA_2 + 2*BETA)
            p[i,:,:,k+1] = (RHO*g(x_in[:,i,:,k+1]) + GAMMA_2*p[i,:,:,k]) / (RHO + GAMMA_2)
            u_in[i,:,:,k+1] = u_in[i,:,:,k] + BETA*(x_in[i,:,:,k+1] - l[i,:,:,k+1])
            u_out[i,:,:,k+1] = u_out[i,:,:,k] + BETA*(x_out[i,:,:,k+1] - y[i,:,:,k+1])

    # Save the results
    timestamp = datetime.now()
    np.savez_compressed(f"{RESULTS_DIR}/{timestamp}.npz", x_in=x_in, x_out=x_out, u_in=u_in, u_out=u_out, p=p, y=y)

    # Residuals inf norm
    res = np.vstack([x_in - x_out.transpose((1,0,2,3)), x_out - x_in.transpose((1,0,2,3))])
    res_inf_norm = np.max(np.abs(res), (0,1,2))

    # res_1_inf_norm = np.max(np.abs(x_in - x_out.transpose((1,0,2,3))), (0,1,2))
    # res_2_inf_norm = np.max(np.abs(x_out - x_in.transpose((1,0,2,3))), (0,1,2))
    # u_inf_norm = np.max(np.abs(u_in - u_out.transpose((1,0,2,3))), (0,1,2))
    
    # c1 = np.max(np.abs(x_in - l), (0,1,2))
    # c2 = np.max(np.abs(x_out - y), (0,1,2))

    # c1 = np.max(np.abs(x_in - Bx_out), (0,1,2))
    # c2 = np.max(np.abs(x_out - Bx_in), (0,1,2))

    # Plotting
    latex_preamble = [
        r'\usepackage{amsfonts}',
        r'\usepackage{amssymb}',
        r'\usepackage{amsmath}',
    ]

    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern"],
        "text.latex.preamble": "".join(latex_preamble)
    })

    fig, ax = plt.subplots(2, figsize=(3.5, 3.5), sharex=True)
    for i in range(2):
        ax[i].xaxis.set_tick_params(labelsize=6)
        ax[i].grid(axis="both", alpha=0.2, lw=0.5)
        ax[i].xaxis.set_tick_params(labelsize=6)
        ax[i].yaxis.set_tick_params(labelsize=6)
        ax[0].axhline(0, c="k", alpha=0.4, lw=0.8)
    ax[-1].set_xlabel("Iterations [$k$]", fontsize=8)

    ax[0].plot(res_inf_norm[:K_PLOT], lw=0.9)  
    ax[0].set_ylabel(
        r'$\left\|\text{col}\left(\begin{bmatrix}'
        r'\hat{\mathbf{x}}_i - \check{\mathbf{x}}_i \\'
        r'\check{\mathbf{x}}_i - \hat{\mathbf{x}}_i'
        r'\end{bmatrix}\right)_{i \in \mathcal{A}}\right\|_\infty$',
        fontsize=6
    )
    
    plt.savefig(f"{PLOTS_DIR}/residuals.pdf", bbox_inches="tight")

    # ax[1].plot(res_2_inf_norm)
    # ax[1].axhline(0, c="k", alpha=0.4)
    # plt.grid(axis="both", alpha=0.2)

    # ax[0].plot(res_1_inf_norm)
    # ax[1].plot(u_inf_norm)
    # ax.plot(res_2_inf_norm)

    plt.show()
        

    


