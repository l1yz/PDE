from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from einops import rearrange
from IPython.display import HTML
from matplotlib.animation import FuncAnimation



from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable


def imshow_movie(sol, frames=250, t=None, interval=100, tight=False, title='', cmap='viridis', aspect='equal', live_cbar=False, save_to=None, show=True):

    fig, ax = plt.subplots()
    div = make_axes_locatable(ax)
    cax = div.append_axes('right', '5%', '5%')

    cv0 = sol[0]
    # Here make an AxesImage rather than contour
    im = ax.imshow(cv0, cmap=cmap, aspect=aspect)
    cb = fig.colorbar(im, cax=cax)
    tx = ax.set_title('Frame 0')
    vmax = np.max(sol)
    vmin = np.min(sol)
    ax.set_xticks([])
    ax.set_yticks([])
    if tight:
        plt.tight_layout()

    def animate(frame):
        arr, t = frame
        im.set_data(arr)
        if live_cbar:
            vmax = np.max(arr)
            vmin = np.min(arr)
            im.set_clim(vmin, vmax)
        tx.set_text(f'{title} t={t:.2f}')

    time, w, h = sol.shape
    if t is None:
        t = np.arange(time)
    inc = max(time//frames, 1)
    sol_frames = sol[::inc]
    t_frames = t[::inc]
    frames = list(zip(sol_frames, t_frames))
    ani = FuncAnimation(fig, animate,
                        frames=frames, interval=interval,)
    plt.close()

    if save_to is not None:
        p = Path(save_to).with_suffix('.gif')
        ani.save(p, writer='pillow', fps=30)

    if show:
        return HTML(ani.to_jshtml())


def line_movie(sol, frames=50, t=None, x=None,  title='', interval=100, ylim=None, save_to=None, show=True, legend=None, tight=False):
    sol = np.asarray(sol)
    if len(sol.shape) == 2:
        sol = np.expand_dims(sol, axis=0)

    n_lines, time, space = sol.shape
    sol = rearrange(sol, 'l t s -> t s l')
    fig, ax = plt.subplots()
    ax.set_ylim([sol.min(), sol.max()])
    if ylim is not None:
        ax.set_ylim(ylim)
    if x is None:
        x = np.arange(sol.shape[1])

    cycler = plt.cycler(
        linestyle=['-', '--']*5, color=plt.rcParams['axes.prop_cycle'].by_key()['color'])
    ax.set_prop_cycle(cycler)
    line = ax.plot(x, sol[0], )
    if tight:
        plt.tight_layout()

    if legend is not None:
        ax.legend(legend)

    def animate(frame):
        sol, t = frame
        ax.set_title(f'{title} t={t:.3f}')
        for i, l in enumerate(line):
            l.set_ydata(sol[:, i])
        return line

    def init():
        line.set_ydata(np.ma.array(x, mask=True))
        return line,

    if t is None:
        t = np.arange(time)
    inc = max(time//frames, 1)
    sol_frames = sol[::inc]
    t_frames = t[::inc]
    sol_frames = sol[::inc]
    frames = list(zip(sol_frames, t_frames))
    ani = FuncAnimation(fig, animate, frames=frames,
                        interval=interval, blit=True)
    plt.close()
    if save_to is not None:
        p = Path(save_to).with_suffix('.gif')
        ani.save(p, writer='pillow', fps=30)

    if show:
        return HTML(ani.to_jshtml())


def trajectory_movie(y, frames=50, title='', ylabel='', xlabel='Time', legend=[],x=None, interval=100, ylim=None, save_to=None):

    y = np.asarray(y)
    if x is None:
        x = np.arange(len(y))

    fig, ax = plt.subplots()  
    total = len(x)
    inc = max(total//frames, 1)
    x = x[::inc]
    y = y[::inc]
    if ylim is None:
        ylim = np.array([y.min(), y.max()])
    xlim = [x.min(), x.max()]

    def animate(i):
        ax.cla()
        ax.plot(x[:i], y[:i], marker='o', markevery=[-1])
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.legend(legend, loc='lower right')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f'{title} t={x[i]:.2f}')

    ani = FuncAnimation(fig, animate, frames=len(x), interval=interval)
    plt.close()

    if save_to is not None:
        p = Path(save_to).with_suffix('.gif')
        ani.save(p, writer='pillow', fps=30)

    return HTML(ani.to_jshtml())


from matplotlib.gridspec import GridSpec

def multi_experiment_trajectory_snapshot(all_phis, times=None, title='', ylabel='', xlabel='Time', 
                                         param_names=None, exp_names=None, colors=None, linestyles=None, 
                                         ylim=None, save_to=None):
    
    n_experiments = len(all_phis)
    n_params = all_phis[0].shape[1]
    
    if times is None:
        times = [np.arange(len(phis)) for phis in all_phis]
    elif not isinstance(times, list):
        times = [times] * n_experiments
    
    fig = plt.figure(figsize=(16, 8))
    gs = GridSpec(1, 2, width_ratios=[3, 1])  # Create a grid with two columns
    
    ax = fig.add_subplot(gs[0])  # Main plot
    leg_ax = fig.add_subplot(gs[1])  # Legend
    leg_ax.axis('off')  # Turn off axis for legend
    
    if ylim is None:
        ylim = np.array([min(phis.min() for phis in all_phis), max(phis.max() for phis in all_phis)])
    xlim = [min(t.min() for t in times), max(t.max() for t in times)]
    
    if colors is None:
        colors = plt.cm.rainbow(np.linspace(0, 1, n_params))
    if linestyles is None:
        linestyles = ['-', '--', ':', '-.'] * (n_experiments // 4 + 1)
    if param_names is None:
        param_names = [f'$\\phi_{{{i}}}$' for i in range(n_params)]
    if exp_names is None:
        exp_names = [f'Exp {i+1}' for i in range(n_experiments)]
    
    for i, (phis, t) in enumerate(zip(all_phis, times)):
        for j in range(n_params):
            ax.plot(t, phis[:, j], color=colors[j], linestyle=linestyles[i])
    
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    
    # Create separate legends for parameters (colors) and experiments (line styles)
    param_legend = [plt.Line2D([0], [0], color=colors[i], lw=2) for i in range(n_params)]
    exp_legend = [plt.Line2D([0], [0], color='gray', linestyle=linestyles[i], lw=2) for i in range(n_experiments)]
    
    # Add the legends to the legend axis
    leg_ax.legend(param_legend, param_names, loc='upper left', title='Parameters')
    leg_ax.legend(exp_legend, exp_names, loc='upper left', bbox_to_anchor=(0, 0.6), title='Experiments')
    
    # Combine both legends
    combined_handles = param_legend + exp_legend
    combined_labels = param_names + exp_names
    #combined_titles = ['Params'] * n_params + ['Experiments'] * n_experiments
    
    # Create a new legend with both parameters and experiments
    leg = leg_ax.legend(combined_handles, combined_labels, loc='center', title=None)
    
    # Add subtitle for each group in the legend
    # for t, l in zip(combined_titles, leg.get_texts()):
    #     l.set_multialignment('left')
    #     l.set_text(f'{t}\n{l.get_text()}')
    
    plt.tight_layout()
    
    if save_to is not None:
        p = Path(save_to).with_suffix('.png')
        plt.savefig(p, dpi=300, bbox_inches='tight')
    
    plt.show()

# def latent_plot_2d(latent, labels, title='', save_to=None): # this takes the predicted latent space and plots its trajectory
#     fig, ax = plt.subplots()
#     ax.plot(latent[:, 0], latent[:, 1],'--', c=labels)
#     ax.set_title(title)
#     if save_to is not None:
#         p = Path(save_to).with_suffix('.png')
#         plt.savefig(p)
#     plt.show()
# # we also plot the 3d dynamics of the latent state when it is 3d
# def latent_plot_3d(latent, labels, title='', save_to=None):
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     ax.plot(latent[:, 0], latent[:, 1], latent[:, 2], '--', c=labels)
#     ax.set_title(title)
#     if save_to is not None:
#         p = Path(save_to).with_suffix('.png')
#         plt.savefig(p)
#     plt.show()


def latent_space_2d(all_phis, times=None, title='', xlabel='$\\phi_1$', ylabel='$\\phi_2$', 
                               exp_names=None, colors=None, linestyles=None, 
                               xlim=None, ylim=None, save_to=None):
    
    n_experiments = len(all_phis)
    
    if times is None:
        times = [np.arange(len(phis)) for phis in all_phis]
    elif not isinstance(times, list):
        times = [times] * n_experiments
    
    fig = plt.figure(figsize=(12, 10))
    gs = GridSpec(1, 2, width_ratios=[3, 1])  # Create a grid with two columns
    
    ax = fig.add_subplot(gs[0])  # Main plot
    leg_ax = fig.add_subplot(gs[1])  # Legend
    leg_ax.axis('off')  # Turn off axis for legend
    
    if xlim is None:
        xlim = np.array([min(phis[:, 0].min() for phis in all_phis), 
                         max(phis[:, 0].max() for phis in all_phis)])
    if ylim is None:
        ylim = np.array([min(phis[:, 1].min() for phis in all_phis), 
                         max(phis[:, 1].max() for phis in all_phis)])
    if colors is None:
        colors = plt.cm.viridis(np.linspace(0, 1, n_experiments))
    if linestyles is None:
        linestyles = ['-', '--', ':', '-.'] * (n_experiments // 4 + 1)
    if exp_names is None:
        exp_names = [f'Exp {i+1}' for i in range(n_experiments)]
    
    for i, (phis, t) in enumerate(zip(all_phis, times)):
        # Create a color gradient for the trajectory
        n_points = len(t)
        color_map = LinearSegmentedColormap.from_list("", ["lightblue", colors[i]])
        color_array = color_map(np.linspace(0, 1, n_points))
        
        # Plot the trajectory with color gradient
        for j in range(n_points - 1):
            ax.plot(phis[j:j+2, 0], phis[j:j+2, 1], color=color_array[j], 
                    linestyle=linestyles[i], linewidth=2)
        
        # Add markers for start and end points
        ax.plot(phis[0, 0], phis[0, 1], 'o', color=colors[i], markersize=8, label=f'{exp_names[i]} Start')
        ax.plot(phis[-1, 0], phis[-1, 1], 's', color=colors[i], markersize=8, label=f'{exp_names[i]} End')
    
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Create legend
    handles, labels = ax.get_legend_handles_labels()
    leg = leg_ax.legend(handles, labels, loc='center', title='Experiments')
    
    plt.tight_layout()
    
    if save_to is not None:
        p = Path(save_to).with_suffix('.png')
        plt.savefig(p, dpi=300, bbox_inches='tight')
    
    plt.show()

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LinearSegmentedColormap

def latent_space_3d(all_phis, times=None, title='', xlabel='$\\phi_1$', ylabel='$\\phi_2$', zlabel='$\\phi_3$',
                               exp_names=None, colors=None, linestyles=None, 
                               xlim=None, ylim=None, zlim=None, save_to=None):
    
    n_experiments = len(all_phis)
    
    if times is None:
        times = [np.arange(len(phis)) for phis in all_phis]
    elif not isinstance(times, list):
        times = [times] * n_experiments
    
    fig = plt.figure(figsize=(15, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    if xlim is None:
        xlim = np.array([min(phis[:, 0].min() for phis in all_phis), 
                         max(phis[:, 0].max() for phis in all_phis)])
    if ylim is None:
        ylim = np.array([min(phis[:, 1].min() for phis in all_phis), 
                         max(phis[:, 1].max() for phis in all_phis)])
    if zlim is None:
        zlim = np.array([min(phis[:, 2].min() for phis in all_phis), 
                         max(phis[:, 2].max() for phis in all_phis)])
    
    if colors is None:
        colors = plt.cm.viridis(np.linspace(0, 1, n_experiments))
    if linestyles is None:
        linestyles = ['-', '--', ':', '-.'] * (n_experiments // 4 + 1)
    if exp_names is None:
        exp_names = [f'Exp {i+1}' for i in range(n_experiments)]
    
    for i, (phis, t) in enumerate(zip(all_phis, times)):
        # Create a color gradient for the trajectory
        n_points = len(t)
        #color_map = LinearSegmentedColormap.from_list("", ["lightblue", colors[i]])
        #color_array = color_map(np.linspace(0, 1, n_points))
        
        # Plot the trajectory with color gradient
        for j in range(n_points - 1):
            ax.plot(phis[j:j+2, 0], phis[j:j+2, 1], phis[j:j+2, 2], color = colors[i],linestyle=linestyles[i], linewidth=2)
        
        # Add markers for start and end points
        ax.scatter(phis[0, 0], phis[0, 1], phis[0, 2], c=[colors[i]], marker='o', s=100, label=f'{exp_names[i]} Start')
        ax.scatter(phis[-1, 0], phis[-1, 1], phis[-1, 2], c=[colors[i]], marker='s', s=100, label=f'{exp_names[i]} End')
    
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_zlim(zlim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    ax.set_title(title)
    
    # Improve 3D viewing angle
    ax.view_init(elev=20, azim=45)
    
    # Create legend
    ax.legend(loc='upper left', bbox_to_anchor=(1.1, 1), title='Experiments')
    
    plt.tight_layout()
    
    if save_to is not None:
        p = Path(save_to).with_suffix('.png')
        plt.savefig(p, dpi=300, bbox_inches='tight')
    
    plt.show()

# from mpl_toolkits.mplot3d import Axes3D
# import ipywidgets as widgets
# from IPython.display import display

def interactive_latent_space_3d(all_phis, times=None, title='', xlabel='$\\phi_1$', ylabel='$\\phi_2$', zlabel='$\\phi_3$',
                                           exp_names=None, colors=None, linestyles=None, 
                                           xlim=None, ylim=None, zlim=None, save_to=None):
    
    n_experiments = len(all_phis)
    
    if times is None:
        times = [np.arange(len(phis)) for phis in all_phis]
    elif not isinstance(times, list):
        times = [times] * n_experiments
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    if xlim is None:
        xlim = np.array([min(phis[:, 0].min() for phis in all_phis), 
                         max(phis[:, 0].max() for phis in all_phis)])
    if ylim is None:
        ylim = np.array([min(phis[:, 1].min() for phis in all_phis), 
                         max(phis[:, 1].max() for phis in all_phis)])
    if zlim is None:
        zlim = np.array([min(phis[:, 2].min() for phis in all_phis), 
                         max(phis[:, 2].max() for phis in all_phis)])
    
    if colors is None:
        colors = plt.cm.viridis(np.linspace(0, 1, n_experiments))
    if linestyles is None:
        linestyles = ['-', '--', ':', '-.'] * (n_experiments // 4 + 1)
    if exp_names is None:
        exp_names = [f'Exp {i+1}' for i in range(n_experiments)]
    
    def update_plot(elev, azim):
        ax.clear()
        for i, (phis, t) in enumerate(zip(all_phis, times)):
            n_points = len(t)
            color_map = LinearSegmentedColormap.from_list("", ["lightblue", colors[i]])
            color_array = color_map(np.linspace(0, 1, n_points))
            
            for j in range(n_points - 1):
                ax.plot(phis[j:j+2, 0], phis[j:j+2, 1], phis[j:j+2, 2], color=color_array[j], 
                        linestyle=linestyles[i], linewidth=2)
            
            ax.scatter(phis[0, 0], phis[0, 1], phis[0, 2], c=[colors[i]], marker='o', s=100, label=f'{exp_names[i]} Start')
            ax.scatter(phis[-1, 0], phis[-1, 1], phis[-1, 2], c=[colors[i]], marker='s', s=100, label=f'{exp_names[i]} End')
        
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_zlim(zlim)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_zlabel(zlabel)
        ax.set_title(title)
        ax.legend(loc='upper left', bbox_to_anchor=(1.1, 1), title='Experiments')
        ax.view_init(elev=elev, azim=azim)
        fig.canvas.draw_idle()
    
    elev_slider = widgets.FloatSlider(value=20, min=0, max=90, step=1, description='Elevation:')
    azim_slider = widgets.FloatSlider(value=45, min=0, max=360, step=1, description='Azimuth:')
    
    widgets.interactive(update_plot, elev=elev_slider, azim=azim_slider)
    
    plt.tight_layout()
    
    if save_to is not None:
        p = Path(save_to).with_suffix('.png')
        plt.savefig(p, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    return fig, ax
