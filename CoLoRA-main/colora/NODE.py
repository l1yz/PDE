'''This is the class of Neural ODEs.
Init by NeuralODE(phi_dim, mu_dim, hidden_dim, depth, key)
'''
import equinox as eqx
import diffrax
import jax

import jax.numpy as jnp
from jax.random import PRNGKey
from jax import random as jr



# Define the ODE function approximator using an MLP
class ODEFunc(eqx.Module):
    mlp: eqx.nn.MLP

    def __init__(self, input_dim, hidden_dim, output_dim, depth, key):
        super().__init__()
        self.mlp = eqx.nn.MLP(input_dim, output_dim, hidden_dim, depth,activation=jax.nn.softplus, key=key)

    def __call__(self, t, y0):
        return self.mlp(y0)

# Define simpler ODE funcs
class ODEFunc_lin(eqx.Module):
    weight: jnp.ndarray
    bias: jnp.ndarray

    def __init__(self, input_dim, output_dim, key):
        super().__init__()
        self.weight = jr.normal(key, (output_dim, input_dim)) * 0.01
        self.bias = jnp.zeros((output_dim,))

    def __call__(self, t, y):
        return jnp.dot(self.weight, y) + self.bias

class ODEFunc_quad(eqx.Module): # mu only affects the linear term
    L: jnp.ndarray
    Q: jnp.ndarray
    bias: jnp.ndarray

    def __init__(self, input_dim, output_dim, key):
        super().__init__()
        key1, key2 = jr.split(key)
        self.L = jr.normal(key1, (output_dim, input_dim-1)) * 0.01
        Q_raw = jr.normal(key2, (output_dim, input_dim-1, input_dim-1)) * 0.01
        # enforce energy conservation       
        Q_symmetric = (Q_raw + jnp.transpose(Q_raw, (0, 2, 1))) / 2
        self.Q = Q_symmetric - (jnp.transpose(Q_symmetric,(1,0,2))+jnp.transpose(Q_symmetric,(1,2,0)))/3
        self.bias = jnp.zeros((output_dim,))

    def __call__(self, t, y):
        mu=y[-1]
        y=y[:-1]
        Ly = jnp.dot(self.L, y)

        Qyy = jnp.einsum('ijk,j,k->i', self.Q, y, y) 

        return  Qyy + mu*Ly + self.bias
    

# class ODEFunc_cubic(eqx.Module):
#     linear_weight: jnp.ndarray
#     quadratic_weight: jnp.ndarray
#     cubic_weight: jnp.ndarray
#     bias: jnp.ndarray

#     def __init__(self, input_dim, output_dim, key):
#         super().__init__()
#         key1, key2, key3 = jr.split(key, 3)
#         self.linear_weight = jr.normal(key1, (output_dim, input_dim)) * 0.01
#         self.quadratic_weight = jr.normal(key2, (output_dim, input_dim, input_dim)) * 0.01
#         self.cubic_weight = jr.normal(key3, (output_dim, input_dim, input_dim, input_dim)) * 0.01
#         self.bias = jnp.zeros((output_dim,))

#     def __call__(self, t, y):
#         linear_term = jnp.dot(self.linear_weight, y)
#         quadratic_term = jnp.einsum('ijk,j,k->i', self.quadratic_weight, y, y)
#         cubic_term = jnp.einsum('ijkl,j,k,l->i', self.cubic_weight, y, y, y)
#         return linear_term + quadratic_term + cubic_term + self.bias
    
# Define the Neural ODE class

# Modified to accept different ODEFunc types
class NODE(eqx.Module):
    ode_func: eqx.Module

    def __init__(self, phi_dim, mu_dim, hidden_dim, depth, keygen, ode_type='mlp'):
        key = PRNGKey(keygen)
        super().__init__()
        
        if ode_type == 'mlp':
            self.ode_func = ODEFunc(phi_dim + mu_dim, hidden_dim, phi_dim, depth, key)
        elif ode_type == 'linear':
            self.ode_func = ODEFunc_lin(phi_dim + mu_dim, phi_dim, key)
        elif ode_type == 'quadratic':
            self.ode_func = ODEFunc_quad(phi_dim + mu_dim, phi_dim, key)
        # elif ode_type == 'cubic': #doesn't work yet
        #     self.ode_func = ODEFunc_cubic(phi_dim + mu_dim, phi_dim, key)
        else:
            raise ValueError(f"Unknown ode_type: {ode_type}")

    def __call__(self, phi0, mu, t_span):
        def func(t, phi, args):
            phi_mu = jnp.concatenate((jnp.array(phi), jnp.array([mu])), axis=0)
            return self.ode_func(t, phi_mu)

        solver = diffrax.Tsit5()

        saveat = diffrax.SaveAt(ts=t_span)

        sol = diffrax.diffeqsolve(
            diffrax.ODETerm(func),
            solver,
            t0=t_span[0],
            t1=t_span[-1],
            dt0=t_span[1] - t_span[0],
            y0=phi0,
            saveat=saveat
        )
        return sol.ys

###############
    
# we use the method of augmented NODE
#  https://arxiv.org/pdf/1904.01681   
# class ANODE(eqx.Module):
#     ode_func: ODEFunc
#     def __init__(self, phi_dim, mu_dim, hidden_dim, aug_dim,depth, keygen):
#         key=PRNGKey(keygen)
#         super().__init__() 
#         self.ode_func = ODEFunc(phi_dim+mu_dim+aug_dim, hidden_dim, phi_dim+aug_dim, depth, key)  # output_dim matches phi_dim

#     def __call__(self, phi0, mu, t_span,aug_dim):
#         def func(t, phi,args):
#             #concatenate phi and mu
#             initial_condition=jnp.concatenate((jnp.array(phi),jnp.zeros(aug_dim), jnp.array([mu])), axis=0)
#             return self.ode_func(t, initial_condition)

#         solver = diffrax.Tsit5()
#         saveat = diffrax.SaveAt(ts=t_span)
#         sol = diffrax.diffeqsolve(
#             diffrax.ODETerm(func),
#             solver,
#             t0=t_span[0],
#             t1=t_span[-1],
#             dt0=t_span[1] - t_span[0],
#             y0=phi0,
#             saveat=saveat
#         )
#         return sol.ys


