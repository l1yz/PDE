### made in one block to test multiple functions 
### with different subsample sizes
import numpy as np
import matplotlib.pyplot as plt
from jax import jit, vmap, grad
import jax.flatten_util
from time import time
import pandas as pd
import jax
import jax.numpy as jnp

sub_sample_sizes = [10, 20, 50, 100, 150, 200, 300, 500]
for j in range(len(sub_sample_sizes)):
    # High Speed Settings
    n_x = 1000  # number of sample points in space
    sub_sample = sub_sample_sizes[j]  # number of parameters to randomly sample
    dt = 5e-3  # time step 
    # set up time and space domain
    Tend = 5.0
    t_eval = jnp.linspace(0.0, Tend, int(Tend/dt) + 1)


    dim = 1
    A, B = 0, 2 * jnp.pi
    x_eval = jnp.expand_dims(jnp.linspace(A, B, n_x), axis=-1)


    from rsng.dnn import build_nn, init_net

    '''
    test with different PDEs and save the plots
    '''

    ############ define a list of test functions with
    # linear advection equations
    def linear_advec1(t, theta):
        u = U(theta, x_eval)
        u_x = U_dx(theta, x_eval)
        c = lambda t :1.0
        return -c(t) * u_x   
    def linear_advec2(t, theta):
        u = U(theta, x_eval)
        u_x = U_dx(theta, x_eval)
        c = lambda t: jnp.sin(5 * t)
        return -c(t) * u_x
    # Allen Cahn
    def allen_cahn1(t, theta):
        u = U(theta, x_eval)
        u_xx = U_ddx(theta, x_eval)
        return 5e-3*u_xx+u-u**3

    def allen_cahn2(t, theta):
        u = U(theta, x_eval)
        u_xx = U_ddx(theta, x_eval)
        return 5e-2*u_xx+u-u**3
    # Burgers
    def burgers1(t, theta):
        u = U(theta, x_eval)
        u_x = U_dx(theta, x_eval)
        u_xx = U_ddx(theta, x_eval)
        return 5e-1*u_xx-u_x*u
    def bburgers2(t, theta):
        u = U(theta, x_eval)
        u_x = U_dx(theta, x_eval)
        u_xx = U_ddx(theta, x_eval)
        return 1e-3*u_xx-u_x*u

    test_funcs = [linear_advec1, linear_advec2, allen_cahn1, allen_cahn2, burgers1, bburgers2]

    for func_i in range(len(test_funcs)):
        key = jax.random.PRNGKey(i)
        width = 25
        depth = 7
        period = 2 * jnp.pi

        net = build_nn(width, depth, period)
        u_scalar, theta_init, unravel = init_net(net, key, dim)

        # used to take gradient and then squeeze
        def gradsqz(f, *args, **kwargs):
            return lambda *fargs, **fkwargs: jnp.squeeze(grad(f, *args, **kwargs)(*fargs, **fkwargs))

        # batch the function over X points
        U = vmap(u_scalar, (None, 0))

        # derivative with respect to theta
        U_dtheta = vmap(grad(u_scalar), (None, 0))

        # first spatial derivative
        U_dx = vmap(gradsqz(u_scalar, 1), (None, 0))

        # second spatial derivatives
        U_ddx = vmap(gradsqz(gradsqz(u_scalar, 1), 1), (None, 0))

        # load the parameters which fit initial condition
        if i <4:
            theta_0 = pd.read_pickle('./rsng/data/theta_init_ac.pkl')
        else:
            theta_0 = pd.read_pickle('./rsng/data/theta_init_burgers.pkl')
        theta_0 = jax.flatten_util.ravel_pytree(theta_0)[0]

        # plot initial condition
        plt.plot(x_eval, U(theta_0, x_eval))
        plt.show()

        rhs=test_funcs[func_i]
        # Use Gaussian to subscale the columns of J
        def rhs_reparameterized(t, theta, key):
            key,_= jax.random.split(key)
            J = U_dtheta(theta, x_eval)  # take the gradient with respect to the parameters
            
            B=jax.random.normal(key,(sub_sample,n_x))
            f = rhs(t, theta)
            # apply B
            J = jnp.dot(B, J)
            f = jnp.dot(B, f)
            theta_dot = jnp.linalg.lstsq(J, f, rcond=1e-4)[0]
            residual = jnp.linalg.norm(jnp.dot(J, theta_dot) - f)
            return theta_dot, residual

        def odeint_rk4(fn, y0, t, key):
            "Adapted from: https://github.com/DifferentiableUniverseInitiative/jax_cosmo/blob/master/jax_cosmo/scipy/ode.py"
            def rk4(carry, t):
                y, t_prev, key = carry
                h = t - t_prev
                key, subkey = jax.random.split(key)

                k1, _ = fn(t_prev, y, subkey)
                k2, _ = fn(t_prev + h / 2, y + h * k1 / 2, subkey)
                k3, _ = fn(t_prev + h / 2, y + h * k2 / 2, subkey)
                k4, _ = fn(t, y + h * k3, subkey)

                y = y + 1.0 / 6.0 * h * (k1 + 2 * k2 + 2 * k3 + k4)
                return (y, t, key), y

            (yf, _, _), y = jax.lax.scan(rk4, (y0, jnp.array(t[0]), key), t)
            return y

        # forward Euler integrator
        def odeint_euler(fn, y0, t, key):
            def euler(carry, t):
                y, t_prev, key = carry
                h = t - t_prev
                key, subkey = jax.random.split(key)

                y_dot, _ = fn(t_prev, y, subkey)
                y = y + h * y_dot
                return (y, t, key), y

            (yf, _, _), y = jax.lax.scan(euler, (y0, jnp.array(t[0]), key), t)
            return y

        def integrate(y0, t):
            return odeint_rk4(rhs_reparameterized, y0, t, key) 

        # here we separate compile time from integration time
        integrate_complied = jit(integrate).lower(theta_0, t_eval).compile() 
        print('jit complied!')
        time_start = time()
        y = integrate_complied(theta_0, t_eval)
        time_end = time()
        print('done!')

        steps = len(t_eval)
        theta_dot = np.zeros((steps, len(x_eval)))
        residuals = []

        for i in range(steps):
            theta = y[i, :]
            theta_dot[i] = jnp.squeeze(U(theta, x_eval))
            _, residual = rhs_reparameterized(t_eval[i], theta, key)
            residual_value = jax.device_get(residual)
            residuals.append(residual_value)
            print(f"Time: {t_eval[i]}, Residual: {residual_value}")

        # Plotting
        plt.imshow(theta_dot, aspect='auto')
        plt.title(f'{test_funcs[i].__name__}_subsample{sub_sample}_sol')
        plt.xlabel('space')
        plt.ylabel('time')
        plt.show()
        # save the plot as png and name it with the function name
        plt.savefig(f'{test_funcs[i].__name__}_subsample{sub_sample}_sol.png')

        # Plot residuals over time
        plt.figure()
        plt.plot(t_eval, residuals)
        plt.xlabel('Time')
        plt.ylabel('Residual')
        plt.title(f'{test_funcs[func_i].__name__}_subsample{sub_sample}_residual')
        plt.show()
        #save the plots as png
        plt.savefig(f'{test_funcs[func_i].__name__}_subsample{sub_sample}_residual.png')


