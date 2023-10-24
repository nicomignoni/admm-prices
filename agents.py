#!/usr/bin/env -S python -i
import numpy as np
import cvxpy as cp


class Agent:
    def __init__(self, N, T, is_neighbour, max_x_in, max_x_out, beta, gamma_1):
        self.is_neighbour = is_neighbour
        self.max_x_in = max_x_in
        self.max_x_out = max_x_out

        self.x_in = cp.Variable((N,T), nonneg=True)
        self.x_out = cp.Variable((N,T), nonneg=True)

        self.old_x_in  = cp.Parameter((N,T), nonneg=True)
        self.old_x_out = cp.Parameter((N,T), nonneg=True)
 
        self.y     = cp.Parameter((N,T))
        self.y_ext = cp.Parameter((N,T))
        self.u_in  = cp.Parameter((N,T))
        self.u_out = cp.Parameter((N,T))
        self.p     = cp.Parameter((N,T), nonneg=True)
        self.p_ext = cp.Parameter((N,T), nonneg=True)

        # Sets a 0 to the variables corresponding to agents not in the neighbourhood
        self.constraints = [
            self.x_in <= max_x_in * np.tile(is_neighbour, [T, 1]).T,
            self.x_out <= max_x_out * np.tile(is_neighbour, [T, 1]).T
        ]
       
        self.phi = cp.sum_squares(self.x_in + self.x_out)
        self.energy_cost = cp.multiply(self.p_ext, self.x_in).sum() - cp.multiply(self.p, self.x_out).sum() 
        self.lagrange = self.phi + self.energy_cost + \
                        cp.multiply(self.u_in, self.x_in).sum() + \
                        cp.multiply(self.u_out, self.x_out).sum() + \
                        0.5*gamma_1 * cp.sum_squares(self.x_in - self.old_x_in) + \
                        0.5*gamma_1 * cp.sum_squares(self.x_out - self.old_x_out) + \
                        0.5*beta * cp.sum_squares(self.x_in - self.y_ext) + \
                        0.5*beta * cp.sum_squares(self.x_out - self.y)
       
                      

class Retailer(Agent):
    def __init__(self, a_in, a_out, N, T, is_neighbour, max_x_in, max_x_out, beta, gamma_1):
        super().__init__(N, T, is_neighbour, max_x_in, max_x_out, beta, gamma_1)
        self.a_in  = a_in
        self.a_out = a_out
        
        self.constraints += [
            self.x_in.sum(axis=0) <= self.a_in,
            self.x_out.sum(axis=0) <= self.a_out,   
        ]

        self.prob = cp.Problem(cp.Minimize(self.lagrange), self.constraints)


class Prosumer(Agent):
    def __init__(self, delta, N, T, is_neighbour, max_x_in, max_x_out, beta, gamma_1):
        super().__init__(N, T, is_neighbour, max_x_in, max_x_out, beta, gamma_1)
        self.delta = delta

        self.constraints += [
            self.x_out.sum(axis=0) - self.x_in.sum(axis=0) == self.delta
        ]

        self.prob = cp.Problem(cp.Minimize(self.lagrange), self.constraints)


class Storage(Agent):
    def __init__(self, s0, s_max, alpha, eta_in, eta_out, N, T, is_neighbour, max_x_in, max_x_out, beta, gamma_1):
        super().__init__(N, T, is_neighbour, max_x_in, max_x_out, beta, gamma_1)
        self.s0 = s0
        self.s_max = s_max
        self.alpha = alpha
        self.eta_in = eta_in 
        self.eta_out = eta_out

        self.s = cp.Variable((N,T), nonneg=True)

        self.constraints += [
            self.s == alpha*cp.hstack([s0[:,None], self.s[:,:-1]]) + eta_in*self.x_in - self.x_out/eta_out,
            self.s.sum(axis=0) <= s_max        
        ]

        self.prob = cp.Problem(cp.Minimize(self.lagrange), self.constraints)

