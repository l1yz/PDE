'''This is the class of Neural ODEs.
Init by NeuralODE(phi_dim, mu_dim, hidden_dim, depth, key)
'''
import equinox as eqx
import jax
import jax.numpy as jnp
from jax.random import PRNGKey
import diffrax

# Define the ODE function approximator using an MLP
class ODEFunc(eqx.Module):
    mlp: eqx.nn.MLP

    def __init__(self, input_dim, hidden_dim, output_dim, depth, key):
        super().__init__()
        self.mlp = eqx.nn.MLP(input_dim, output_dim, hidden_dim, depth,activation=jax.nn.relu, key=key)

    def __call__(self, t, phi_mu):
        # Make sure mu is correctly broadcasted
        #mu_expanded = jnp.broadcast_to(mu, (phi.shape[0], 1))  # Ensure mu is repeated for each batch of phi
        inputs = phi_mu
        outputs=self.mlp(inputs)
        # value_outputs=jax.device_get(outputs)
        # print("MLP output at time", t, ":", value_outputs)

        return outputs

# Define the Neural ODE class

class NODE(eqx.Module):
    ode_func: ODEFunc

    def __init__(self, phi_dim, mu_dim, hidden_dim, depth, keygen):
        key=PRNGKey(keygen)
        super().__init__()
        self.ode_func = ODEFunc(phi_dim+mu_dim, hidden_dim, phi_dim, depth, key)  # output_dim matches phi_dim

    def __call__(self, phi0, mu, t_span):
        def func(t, phi, args):
            #concatenate phi and mu
            phi_mu=jnp.concatenate((jnp.array(phi), jnp.array([mu])), axis=0)
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

# Quick example to check the shape

