
from typing import Callable, Optional

import flax.linen as nn
import jax.numpy as jnp
from flax.linen import initializers


class Periodic(nn.Module):

    width: int
    period: Optional[jnp.ndarray]
    param_dtype = jnp.float32
    with_bias: bool = True
    w_init: Callable = initializers.lecun_normal()

    @nn.compact
    def __call__(self, x):
        dim, f = x.shape[-1], self.width
        w_init = self.w_init
        period = jnp.asarray(self.period)

        a = self.param('a', w_init, (f, dim), self.param_dtype)
        phi = self.param('c', w_init, (f, dim), self.param_dtype)

        omeg = jnp.pi*2/period
        o = a*jnp.cos(omeg*x+phi)
        if self.with_bias:
            b = self.param('b', w_init, (f, dim), self.param_dtype)
            o += b

        o = jnp.mean(o, axis=1)

        return o


class CoLoRA(nn.Module):

    width: int
    rank: int
    full: bool
    w_init: Callable = initializers.lecun_normal()
    b_init: Callable = initializers.zeros_init()
    with_bias: bool = True
    param_dtype = jnp.float32

    @nn.compact
    def __call__(self, X):
        D, K, r = X.shape[-1], self.width, self.rank

        w_init = self.w_init
        b_init = self.b_init
        z_init = initializers.zeros_init()

        W = self.param('W', w_init, (D, K), self.param_dtype)
        A = self.param('A', w_init, (D, r), self.param_dtype)
        B = self.param('B', z_init, (r, K), self.param_dtype)

        if self.full:
            n_alpha = self.rank
        else:
            n_alpha = 1

        alpha = self.param('alpha', z_init, (n_alpha,), self.param_dtype)

        AB = (A*alpha)@B
        AB = AB  # / r
        W = (W + AB)

        out = X@W

        if self.with_bias:
            b = self.param("b", b_init, (K,))
            b = jnp.broadcast_to(b, out.shape)
            out += b

        return out

# from typing import Callable, Optional
# from jax.experimental.ode import odeint

# class NODE(nn.Module):
#     width: int
#     param_dtype = jnp.float32
#     w_init: Callable = initializers.lecun_normal()

#     @nn.compact
#     def __call__(self, t, state, params):
#         # t: current time, state: current value of phi, params: mu
#         concatenated_input = jnp.concatenate([state, params, jnp.array([t])])
#         for i in range(3):  # Simple example with 3 layers
#             concatenated_input = nn.Dense(self.width, kernel_init=self.w_init)(concatenated_input)
#             concatenated_input = jax.nn.relu(concatenated_input)
#         output = nn.Dense(state.shape[0], kernel_init=self.w_init)(concatenated_input)
#         return output


# class NODE(nn.Module):
#     width: int
#     activation: Callable = nn.tanh

#     @nn.compact
#     def __call__(self, phi, t, psi):
#         # Define the neural network for the ODE system
#         x = jnp.concatenate([phi, psi], axis=-1)
#         x = nn.Dense(self.width, dtype=jnp.float32, kernel_init=initializers.lecun_normal())(x)
#         x = self.activation(x)
#         x = nn.Dense(phi.shape[-1], dtype=jnp.float32, kernel_init=initializers.lecun_normal())(x)
#         return x

#     @staticmethod
#     def ode_system(phi, t, params, psi):
#         net = NeuralODE.apply({'params': params}, phi, t, psi)
#         return net

#     def integrate(self, phi_init, psi, t_span):
#         params = self.param('params', initializers.lecun_normal(), (self.width, ))
#         phi_trajectory = odeint(self.ode_system, phi_init, t_span, params, psi)
#         return phi_trajectory