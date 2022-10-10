# functions helpful for plotting discrete/continuous states over time
import wandb
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np
from twarhmm import Posterior
from tqdm.auto import trange
import datetime

import seaborn as sns

sns.set_style("white")
sns.set_context("talk")

color_names = ["windows blue",
               "red",
               "amber",
               "faded green",
               "dusty purple",
               "orange",
               "clay",
               "pink",
               "greyish",
               "mint",
               "cyan",
               "steel blue",
               "forest green",
               "pastel purple",
               "salmon",
               "dark brown"]

colors = sns.xkcd_palette(color_names)
cmap = ListedColormap(colors)

def plot_discrete_latent_states(states_z, K):
    plt.figure(figsize=(10,2))
    cmap_limited = ListedColormap(colors[0:K])
    plt.imshow(states_z[None,:],  aspect="auto", cmap=cmap_limited)
    plt.title("Simulated Discrete Latent States")
    plt.yticks([])
    plt.xlabel("Time")
    plt.show()
    
def remove_frame(ax_array,all_off=False):
    for ax in np.ravel(ax_array):
        if not all_off:
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.yaxis.set_ticks_position('left')
            ax.xaxis.set_ticks_position('bottom')
        if all_off:
            ax.set_axis_off()
    
def plot_discrete_and_continuous_latent_states(states_z, states_x, K,colors = None, fig=None, ax=None):
    
    latent_dim = states_x.shape[1]
    T = states_x.shape[0]
    lim = abs(states_x).max()

    if ax == None:
        fig, ax =plt.subplots(1,1,figsize=(10,2))
    ax.imshow(states_z[None,:],  aspect="auto", cmap=colors, extent=[0, T, -lim, lim*latent_dim], alpha=.7)
    ax.set_yticks([])


    # Plot the continuous latent states


    for d in range(latent_dim):
        ax.plot(states_x[:, d] + lim * d, 'k', lw=1)
    ax.set_yticks(np.arange(latent_dim) * lim, ["$x_{}$".format(d+1) for d in range(latent_dim)])
    ax.set_xticks([])
    ax.set_xlim(0, T)
    
    
    return fig
    
def plot_continuous_states_and_emissions(states_x, emissions):
    plt.figure(figsize=(10, 6))
    emissions_dim = emissions.shape[1]
    latent_dim = states_x.shape[1]
    T = states_x.shape[0]
    gs = plt.GridSpec(2, 1, height_ratios=(1, emissions_dim/latent_dim))

    # Plot the continuous latent states
    lim = abs(states_x).max()
    plt.subplot(gs[0])
    for d in range(latent_dim):
        plt.plot(states_x[:, d] + lim * d, '-k')
    plt.yticks(np.arange(latent_dim) * lim, ["$x_{}$".format(d+1) for d in range(latent_dim)])
    plt.xticks([])
    plt.xlim(0, T)
    plt.title("Simulated Latent States")

    lim = abs(emissions).max()
    plt.subplot(gs[1])
    for n in range(emissions_dim):
        plt.plot(emissions[:, n] - lim * n, '-')
    plt.yticks(-np.arange(emissions_dim) * lim, ["$y_{{ {} }}$".format(n+1) for n in range(emissions_dim)])
    plt.xlabel("time")
    plt.xlim(0, T)

    plt.title("Simulated emissions")
    plt.tight_layout()
    
def plot_vector_field(*args, color='black'):
    num_plots = len(args)
    fig, ax = plt.subplots(1,num_plots,figsize=(4*num_plots, 4))
    ax = np.atleast_1d(ax)
    xlims = [-2, 2]
    ylims = [-2, 2]
    X1, X2 = np.meshgrid(np.linspace(xlims[0], xlims[1], 10), np.linspace(ylims[0], ylims[1], 10))
    points = np.stack((X1, X2))
    for i, A in enumerate(args):
        AX = np.einsum('ij,jkl->ikl', A, points)
#         Q = ax[i].quiver(X1, X2, AX[0] - X1, AX[1] - X2, units='width', color=plt.cm.viridis_r(i/(len(args)-1)),scale = 2,scale_units = 'xy')  
        Q = ax[i].quiver(X1, X2, AX[0] - X1, AX[1] - X2, units='width', color=color,scale = 2,scale_units = 'xy') 
        ax[i].set_xlim(xlims)
        ax[i].set_ylim(ylims)
#         ax[i].scatter(0,0,color=plt.cm.viridis_r(i/(len(args)-1)),s = 10)
        ax[i].scatter(0,0,color=color,s = 10)

    return ax    

def plot_continuous_latent_states(states_x, var_x=None, title="", spacing=1):
    plt.figure(figsize=(10, 6))
    T = states_x.shape[0]
    latent_dim = states_x.shape[1]
    gs = plt.GridSpec(.1, 1)
    # Plot the continuous latent states
    if var_x is not None:
        lim = abs(states_x + np.sqrt(var_x)).max()
    else:
        lim = abs(states_x).max()
    lim *= spacing
    plt.subplot(gs[0])
    for d in range(latent_dim):
        x = states_x[:, d] + lim * d
        plt.plot(x, '-k')
        if var_x is not None:
            plt.fill_between(np.arange(x.shape[0]), x+np.sqrt(var_x[:, d]), x-np.sqrt(var_x[:, d]))
    plt.yticks(np.arange(latent_dim) * lim, ["$x_{}$".format(d+1) for d in range(latent_dim)])
    plt.xticks([])
    plt.xlim(0, T)
    plt.title(title)

def wnb_histogram_plot(posteriors, tau_duration=False,duration_plot=False, state_usage_plot=False, ordered_state_usage=False, state_switch=False):
    if tau_duration + state_usage_plot + ordered_state_usage + state_switch == 0:
        print('no histogram selected!')
        return

    data_dim = posteriors[0].model.data_dim
    total_states = posteriors[0].model.observations.num_states * len(posteriors[0].model.observations.taus)

    states = np.concatenate([posterior.get_states() for posterior in posteriors])
    state_usage = np.bincount(states, minlength=total_states)

    num_taus = len(posteriors[0].model.taus)

    if tau_duration:
        durations = Posterior.state_durations(states, total_states)
        duration_mean = np.mean(np.concatenate(durations)* 1000 / 30)
        wandb.log({'duration_mean': duration_mean})
        print('duration mean: ', duration_mean)
        duration_means = np.zeros(total_states)
        duration_covs = np.zeros(total_states)
        for k in range(len(durations)):
            duration_means[k] = durations[k].sum()
            duration_covs[k] = np.std(durations[k])

        fig, ax_array = plt.subplots(4, int(np.ceil(posteriors[0].model.observations.num_states/4)), sharex=True, sharey=True)
        for k in range(posteriors[0].model.observations.num_states):
            ax = np.ravel(ax_array)[k]
            ax.bar(np.arange(num_taus), duration_means[k * num_taus:k * num_taus + num_taus] * 1000 / 30)
            # ax.set_xticks([0, 1, 2, 3, 4])
            # ax.set_xticklabels([1, 2, 3, 4, 5])
            ax.tick_params(axis='x', which='major', labelsize=12)
            ax.tick_params(axis='y', which='major', labelsize=12)
            ax.set_title(k, fontdict={'fontsize': 12})

        fig.add_subplot(111, frameon=False)
        # hide tick and tick label of the big axis
        plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
        plt.xlabel("Tau", labelpad=5)
        plt.ylabel("Amount of time in each state (ms)", labelpad=15)
        plt.tight_layout()
        wandb.log({'tau_distribution_plot':wandb.Image(fig)}, commit=True)

    if duration_plot:
        fig, ax = plt.subplots()
        durations = Posterior.state_durations(states, total_states)
        durations = np.concatenate(durations) * 1000 / 30
        plt.hist(durations, bins=50)
        plt.xlabel("Duration (ms)")
        plt.ylabel("Count")
        plt.tight_layout()
        wandb.log({'duration_plot': wandb.Image(fig)}, commit=True)

    if state_usage_plot:
        fig, ax = plt.subplots()
        plt.bar(np.arange(total_states), state_usage)
        plt.xlabel("state index")
        plt.ylabel("num frames")
        plt.title("histogram of inferred state usage")
        plt.tight_layout()
        wandb.log({'state_usage':wandb.Image(fig)}, commit=True)

    if ordered_state_usage:
        fig, ax = plt.subplots()
        order = np.argsort(state_usage / state_usage.sum())[::-1]
        plt.bar(np.arange(total_states), (state_usage / state_usage.sum())[order])
        plt.xlabel("state index [ordered]")
        plt.ylabel("frequency")
        plt.title("ordered histogram of inferred state usage")
        plt.tight_layout()
        wandb.log({'ordered_state_usage':wandb.Image(fig)}, commit=True)

    if state_switch:
        fig, ax = plt.subplots()
        changepoint_states = np.concatenate([posterior.state_switch() for posterior in posteriors])
        changepoint_usage = np.bincount(changepoint_states, minlength=total_states)
        plt.bar(np.arange(total_states), changepoint_usage)
        plt.xlabel("state index")
        plt.ylabel("number of switches")
        plt.title("histogram of state switches")
        plt.tight_layout()
        wandb.log({'state_switch': wandb.Image(fig)}, commit=True)


def wnb_histogram_plot_cont(posteriors, duration_plot=False, state_usage_plot=False, ordered_state_usage=False, state_switch=False):
    if  duration_plot + state_usage_plot + ordered_state_usage + state_switch == 0:
        print('no histogram selected!')
        return

    data_dim = posteriors[0].model.data_dim
    total_states = posteriors[0].model.observations.num_states

    states = np.concatenate([posterior.get_states() for posterior in posteriors])
    state_usage = np.bincount(states, minlength=total_states)

    durations = Posterior.state_durations(states, total_states)
    wandb.log({'duration_mean': np.mean(np.concatenate(durations) * 1000 / 30)})

    if duration_plot:
        fig, ax = plt.subplots()
        durations = Posterior.state_durations(states, total_states)
        durations = np.concatenate(durations) * 1000 / 30
        plt.hist(durations, bins=50)
        plt.xlabel("Duration (ms)")
        plt.ylabel("Count")
        plt.tight_layout()
        wandb.log({'duration_plot': wandb.Image(fig)}, commit=True)

    if state_usage_plot:
        fig, ax = plt.subplots()
        plt.bar(np.arange(total_states), state_usage)
        plt.xlabel("state index")
        plt.ylabel("num frames")
        plt.title("histogram of inferred state usage")
        plt.tight_layout()
        wandb.log({'state_usage':wandb.Image(fig)}, commit=True)

    if ordered_state_usage:
        fig, ax = plt.subplots()
        order = np.argsort(state_usage / state_usage.sum())[::-1]
        plt.bar(np.arange(total_states), (state_usage / state_usage.sum())[order])
        plt.xlabel("state index [ordered]")
        plt.ylabel("frequency")
        plt.title("ordered histogram of inferred state usage")
        plt.tight_layout()
        wandb.log({'ordered_state_usage':wandb.Image(fig)}, commit=True)

    if state_switch:
        fig, ax = plt.subplots()
        changepoint_states = np.concatenate([posterior.state_switch() for posterior in posteriors])
        changepoint_usage = np.bincount(changepoint_states, minlength=total_states)
        plt.bar(np.arange(total_states), changepoint_usage)
        plt.xlabel("state index")
        plt.ylabel("number of switches")
        plt.title("histogram of state switches")
        plt.tight_layout()
        wandb.log({'state_switch': wandb.Image(fig)}, commit=True)

# for plotting PCs associated with each state and tau
def plot_multiple_average_pcs(state_list,
                              data_dim,
                              posteriors,
                              spc=4,
                              pad=30):
    '''
    '''
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    cmap = plt.get_cmap('viridis')
    colorlist = [i / len(state_list) for i in range(len(state_list))]
    # print(colorlist)

    for color, state_idx in enumerate(state_list):

        # Find slices for this state
        slices = extract_syllable_slices(state_idx, posteriors, num_instances=20000)
        # Find maximum duration
        durs = []
        num_slices = 0
        for these_slices in slices:
            for slc in these_slices:
                durs.append(slc.stop - slc.start)
                num_slices += 1
        if num_slices == 0:
            print("no valid syllables found for state", state_idx)
            return
        #TODO: fix this for when not all taus are used
        max_dur = np.max(durs)
        # Initialize timestamps
        times = np.arange(-pad, max_dur + pad) / 30
        exs = np.nan * np.ones((num_slices, 2 * pad + max_dur, data_dim))
        counter = 0
        # Make figure

        for these_slices, posterior in zip(slices, posteriors):
            data = posterior.data
            for slc in these_slices:
                lpad = min(pad, slc.start)
                rpad = min(pad, len(data['data']) - slc.stop)
                dur = slc.stop - slc.start
                padded_slc = slice(slc.start - lpad, slc.stop + rpad)
                x = data['data'][padded_slc][:, :data_dim]
                exs[counter][(pad - lpad):(pad - lpad + len(x))] = x
                counter += 1
                # Plot single example
                # ax.plot(times[(pad - lpad):(pad - lpad + len(x))],
                #         x - spc * np.arange(data_dim),
                #         ls='-', lw=.5, color='k')
        # take the mean and standard deviation
        ex_mean = np.nanmean(exs, axis=0)
        ex_std = np.nanstd(exs, axis=0)
        for d in range(data_dim):
            # ax.fill_between(times,
            #                 ex_mean[:, d] - 2 * ex_std[:, d] - spc * d,
            #                 ex_mean[:, d] + 2 * ex_std[:, d] - spc * d,
            #                 color='k', alpha=0.25)
            if d == data_dim-1: ax.plot(times[:-1], np.abs(np.diff(ex_mean[:, d])) - spc * d, c=cmap(color / len(state_list)), lw=2,label=color)
            else: ax.plot(times[:-1], np.abs(np.diff(ex_mean[:, d])) - spc * d, c=cmap(color / len(state_list)), lw=2)
        ax.plot([0, 0], [-spc * data_dim, spc], '-r', lw=2, ls='--')
        ax.set_yticks(-spc * np.arange(data_dim))
        ax.set_yticklabels(np.arange(data_dim) + 1)
        ax.set_ylim(-spc * data_dim, spc)
        ax.set_ylabel("principal component")
        # ax.set_xlim(times[0], times[-1])
        ax.set_xlim(-.25, .5)
        ax.set_xlabel("$\Delta t$ [s]")
        num_taus = len(posterior.model.taus)
        ax.set_title("Average PCs for State {}".format(((state_idx - num_taus + 1) / num_taus)))


def centroid_velocity_plot(posteriors):
    speeds = []
    state_list = []
    for posterior in posteriors:
        centroid_x = posterior.data['centroid_x_px']
        centroid_y = posterior.data['centroid_y_px']
        position = np.vstack((centroid_x,centroid_y)).T
        speed = np.linalg.norm(np.diff(position,axis=0),axis=1)
        speeds.append(np.hstack((0,speed)))

        states = posterior.get_states()
        state_list.append(states)

    speeds = np.concatenate(speeds)
    state_list = np.concatenate(state_list)

    num_taus = len(posteriors[0].model.taus)
    total_states = posteriors[0].model.num_discrete_states * num_taus
    fig, ax_array = plt.subplots(4, int(np.ceil(posteriors[0].model.num_discrete_states / 4)), sharex=True, sharey=True)
    for k in range(posteriors[0].model.num_discrete_states):
        avg_state_speeds = np.zeros(num_taus)
        avg_state_vars = np.zeros(num_taus)
        state_tau_list = np.arange(k*num_taus, (k+1)*num_taus)
        for i, state in enumerate(state_tau_list):
            inds = state_list == state
            state_speeds = speeds[inds]
            avg_state_speeds[i] = np.mean(state_speeds)
            avg_state_vars[i] = np.std(state_speeds)
        ax = np.ravel(ax_array)[k]
        ax.bar(np.arange(num_taus), avg_state_speeds)
        ax.set_xticks([i for i in range(num_taus)])
        ax.set_xticklabels([i+1 for i in range(num_taus)])
        ax.tick_params(axis='x', which='major', labelsize=12)
        ax.tick_params(axis='y', which='major', labelsize=12)
        ax.set_title(k, fontdict={'fontsize': 12})

    plt.tight_layout()
    wandb.log({'centroid_velocity': wandb.Image(fig)}, commit=True)

def ave_height_plot(posteriors):
    heights = []
    state_list = []
    for posterior in posteriors:
        height_ave = posterior.data['height_ave_mm']
        heights.append(height_ave)

        states = posterior.get_states()
        state_list.append(states)

    heights = np.concatenate(heights)
    state_list = np.concatenate(state_list)

    num_taus = len(posteriors[0].model.taus)
    total_states = posteriors[0].model.num_discrete_states * num_taus
    fig, ax_array = plt.subplots(4, int(np.ceil(posteriors[0].model.num_discrete_states / 4)), sharex=True, sharey=True)
    for k in range(posteriors[0].model.num_discrete_states):
        max_state_heights = np.zeros(num_taus)
        state_tau_list = np.arange(k*num_taus, (k+1)*num_taus)
        for i, state in enumerate(state_tau_list):
            inds = state_list == state
            state_heights = heights[inds]
            max_state_heights[i] = np.mean(state_heights)
        ax = np.ravel(ax_array)[k]
        ax.bar(np.arange(num_taus), max_state_heights)
        ax.set_xticks([i for i in range(num_taus)])
        ax.set_xticklabels([i+1 for i in range(num_taus)])
        ax.tick_params(axis='x', which='major', labelsize=12)
        ax.tick_params(axis='y', which='major', labelsize=12)
        ax.set_title(k, fontdict={'fontsize': 12})

    plt.tight_layout()
    #plt.savefig('average_height_plot_different-sweep')
    plt.show()

    #wandb.log({'avg_max_height': wandb.Image(fig)}, commit=True)

def save_videos_wandb(posteriors):
    for i in trange(posteriors[0].model.num_discrete_states):
        try:
            filename = "crowd{}_grouped.mp4".format(i)
            # video = make_crowd_movie(i*posteriors[0].num_taus + posteriors[0].num_taus//2, posteriors)
            video = make_crowd_movie(i, posteriors)
            video = video.transpose([0,3,1,2])
            wandb.log(
                {filename: wandb.Video(video, fps=30, format="mp4")})
        except:
            print("failed to create a movie for state", i)

#------------------------------------------------------------------------------------------------------------------------
# functions helpful for animating crowd movies for mouse dataset

import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
import torch
# Specify that we want our tensors on the GPU and in float32
device = torch.device('cpu')
dtype = torch.float64

# Helper function to convert between numpy arrays and tensors
to_t = lambda array: torch.tensor(array, device=device, dtype=dtype)
from_t = lambda tensor: tensor.to("cpu").detach().numpy()

import cv2
from matplotlib import animation
from IPython.display import HTML
from tempfile import NamedTemporaryFile
import base64
import seaborn as sns

sns.set_context("notebook")

# initialize a color palette for plotting
palette = sns.xkcd_palette(["windows blue",
                            "red",
                            "medium green",
                            "dusty purple",
                            "greyish",
                            "orange",
                            "amber",
                            "clay",
                            "pink"])


def sum_tuples(a, b):
    assert a or b
    if a is None:
        return b
    elif b is None:
        return a
    else:
        return tuple(ai + bi for ai, bi in zip(a, b))


_VIDEO_TAG = """<video controls>
 <source src="data:video/x-m4v;base64,{0}" type="video/mp4">
 Your browser does not support the video tag.
</video>"""


def _anim_to_html(anim, fps=20):
    # todo: todocument
    if not hasattr(anim, '_encoded_video'):
        with NamedTemporaryFile(suffix='.mp4') as f:
            anim.save(f.name, fps=fps, extra_args=['-vcodec', 'libx264'])
            video = open(f.name, "rb").read()
        anim._encoded_video = base64.b64encode(video)

    return _VIDEO_TAG.format(anim._encoded_video.decode('ascii'))


def _display_animation(anim, fps=30, start=0, stop=None):
    plt.close(anim._fig)
    return HTML(_anim_to_html(anim, fps=fps))


def play(movie, fps=30, speedup=1, fig_height=6,
         filename=None, show_time=False, show=True):
    # First set up the figure, the axis, and the plot element we want to animate
    T, Py, Px = movie.shape[:3]
    fig, ax = plt.subplots(1, 1, figsize=(fig_height * Px / Py, fig_height))
    im = plt.imshow(movie[0], interpolation='None', cmap=plt.cm.gray)

    if show_time:
        tx = plt.text(0.75, 0.05, 't={:.3f}s'.format(0),
                      color='white',
                      fontdict=dict(size=12),
                      horizontalalignment='left',
                      verticalalignment='center',
                      transform=ax.transAxes)
    plt.axis('off')

    def animate(i):
        im.set_data(movie[i * speedup])
        if show_time:
            tx.set_text("t={:.3f}s".format(i * speedup / fps))
        return im,

        # call the animator.  blit=True means only re-draw the parts that have changed.

    anim = animation.FuncAnimation(fig, animate,
                                   frames=T // speedup,
                                   interval=1,
                                   blit=True)
    plt.close(anim._fig)

    # save to mp4 if filename specified
    if filename is not None:
        with open(filename, "wb") as f:
            anim.save(f.name, fps=fps, extra_args=['-vcodec', 'libx264'])

    # return an HTML video snippet
    if show:
        print("Preparing animation. This may take a minute...")
        return HTML(_anim_to_html(anim, fps=30))


def plot_data_and_states(data, states,
                         spc=4, slc=slice(0, 900),
                         title=None):
    times = data["times"][slc]
    labels = data["labels"][slc]
    x = data["data"][slc]
    num_timesteps, data_dim = x.shape

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.imshow(states[None, slc],
              cmap="cubehelix", aspect="auto",
              extent=(0, times[-1] - times[0], -data_dim * spc, spc))

    ax.plot(times - times[0],
            x - spc * np.arange(data_dim),
            ls='-', lw=3, color='w')
    ax.plot(times - times[0],
            x - spc * np.arange(data_dim),
            ls='-', lw=2, color=palette[0])

    ax.set_yticks(-spc * np.arange(data_dim))
    ax.set_yticklabels(np.arange(data_dim))
    ax.set_ylabel("principal component")
    ax.set_xlim(0, times[-1] - times[0])
    ax.set_xlabel("time [ms]")

    if title is None:
        ax.set_title("data and discrete states")
    else:
        ax.set_title(title)


def extract_syllable_slices(state_idx,
                            posteriors,
                            pad=30,
                            num_instances=50,
                            min_duration=5,
                            max_duration=45,
                            seed=0):
    # Find all the start indices and durations of specified state
    all_mouse_inds = []
    all_starts = []
    all_durations = []
    for mouse, posterior in enumerate(posteriors):
        states = np.argmax(posterior.expected_states(), axis=1)#//posteriors[0].num_taus
        states = np.concatenate([[-1], states, [-1]])
        starts = np.where((states[1:] == state_idx) \
                          & (states[:-1] != state_idx))[0]
        stops = np.where((states[:-1] == state_idx) \
                         & (states[1:] != state_idx))[0]
        durations = stops - starts
        assert np.all(durations >= 1)
        all_mouse_inds.append(mouse * np.ones(len(starts), dtype=int))
        all_starts.append(starts)
        all_durations.append(durations)

    all_mouse_inds = np.concatenate(all_mouse_inds)
    all_starts = np.concatenate(all_starts)
    all_durations = np.concatenate(all_durations)

    # Throw away ones that are too short or too close to start.
    # TODO: also throw away ones close to the end
    valid = (all_durations >= min_duration) \
            & (all_durations < max_duration) \
            & (all_starts > pad)

    num_valid = np.sum(valid)
    all_mouse_inds = all_mouse_inds[valid]
    all_starts = all_starts[valid]
    all_durations = all_durations[valid]

    # Choose a random subset to show
    rng = npr.RandomState(seed)
    subset = rng.choice(num_valid,
                        size=min(num_valid, num_instances),
                        replace=False)

    all_mouse_inds = all_mouse_inds[subset]
    all_starts = all_starts[subset]
    all_durations = all_durations[subset]

    # Extract slices for each mouse
    slices = []
    for mouse in range(len(posteriors)):
        is_mouse = (all_mouse_inds == mouse)
        slices.append([slice(start, start + dur) for start, dur in
                       zip(all_starts[is_mouse], all_durations[is_mouse])])

    return slices

def extract_syllable_slices_indiv(state_idx,
                            posteriors,
                            pad=30,
                            num_instances=50,
                            min_duration=5,
                            max_duration=45,
                            seed=0):
    # Find all the start indices and durations of specified state
    all_mouse_inds = []
    all_starts = []
    all_durations = []
    for mouse, posterior in enumerate(posteriors):
        states = np.argmax(posterior.expected_states(), axis=1)
        states = np.concatenate([[-1], states, [-1]])
        starts = np.where((states[1:] == state_idx) \
                          & (states[:-1] != state_idx))[0]
        stops = np.where((states[:-1] == state_idx) \
                         & (states[1:] != state_idx))[0]
        durations = stops - starts
        assert np.all(durations >= 1)
        all_mouse_inds.append(mouse * np.ones(len(starts), dtype=int))
        all_starts.append(starts)
        all_durations.append(durations)

    all_mouse_inds = np.concatenate(all_mouse_inds)
    all_starts = np.concatenate(all_starts)
    all_durations = np.concatenate(all_durations)

    # Throw away ones that are too short or too close to start.
    # TODO: also throw away ones close to the end
    valid = (all_durations >= min_duration) \
            & (all_durations < max_duration) \
            & (all_starts > pad)

    num_valid = np.sum(valid)
    all_mouse_inds = all_mouse_inds[valid]
    all_starts = all_starts[valid]
    all_durations = all_durations[valid]

    # Choose a random subset to show
    rng = npr.RandomState(seed)
    subset = rng.choice(num_valid,
                        size=min(num_valid, num_instances),
                        replace=False)

    all_mouse_inds = all_mouse_inds[subset]
    all_starts = all_starts[subset]
    all_durations = all_durations[subset]

    # Extract slices for each mouse
    slices = []
    for mouse in range(len(posteriors)):
        is_mouse = (all_mouse_inds == mouse)
        slices.append([slice(start, start + dur) for start, dur in
                       zip(all_starts[is_mouse], all_durations[is_mouse])])

    return slices

def make_crowd_movie(state_idx,
                     posteriors,
                     pad=30,
                     raw_size=(512, 424),
                     crop_size=(80, 80),
                     offset=(50, 50),
                     scale=.5,
                     min_height=10,
                     **kwargs):
    '''
    Adapted from https://github.com/dattalab/moseq2-viz/blob/release/moseq2_viz/viz.py

    Creates crowd movie video numpy array.
    Parameters
    ----------
    dataset (list of dicts): list of dictionaries containing data
    slices (np.ndarray): video slices of specific syllable label
    pad (int): number of frame padding in video
    raw_size (tuple): video dimensions.
    frame_path (str): variable to access frames in h5 file
    crop_size (tuple): mouse crop size
    offset (tuple): centroid offsets from cropped videos
    scale (int): mouse size scaling factor.
    min_height (int): minimum max height from floor to use.
    kwargs (dict): extra keyword arguments
    Returns
    -------
    crowd_movie (np.ndarray): crowd movie for a specific syllable.
    '''
    slices = extract_syllable_slices(state_idx, posteriors)

    xc0, yc0 = crop_size[1] // 2, crop_size[0] // 2
    xc = np.arange(-xc0, xc0 + 1, dtype='int16')
    yc = np.arange(-yc0, yc0 + 1, dtype='int16')

    durs = []
    for these_slices in slices:
        for slc in these_slices:
            durs.append(slc.stop - slc.start)

    if len(durs) == 0:
        print("no valid syllables found for state", state_idx)
        return
    max_dur = np.max(durs)

    # Initialize the crowd movie
    crowd_movie = np.zeros((max_dur + pad * 2, raw_size[1], raw_size[0], 3),
                           dtype='uint8')

    for these_slices, posterior in zip(slices, posteriors):
        data = posterior.data
        for slc in these_slices:
            lpad = min(pad, slc.start)
            rpad = min(pad, len(data['frames']) - slc.stop)
            dur = slc.stop - slc.start
            padded_slc = slice(slc.start - lpad, slc.stop + rpad)
            centroid_x = data['centroid_x_px'][padded_slc] + offset[0]
            centroid_y = data['centroid_y_px'][padded_slc] + offset[1]
            angles = np.rad2deg(data['angles'][padded_slc])
            frames = (data['frames'][padded_slc] / scale).astype('uint8')
            flips = np.zeros(angles.shape, dtype='bool')

            for i in range(lpad + dur + rpad):
                if np.any(np.isnan([centroid_x[i], centroid_y[i]])):
                    continue

                rr = (yc + centroid_y[i]).astype('int16')
                cc = (xc + centroid_x[i]).astype('int16')

                if (np.any(rr < 1)
                        or np.any(cc < 1)
                        or np.any(rr >= raw_size[1])
                        or np.any(cc >= raw_size[0])
                        or (rr[-1] - rr[0] != crop_size[0])
                        or (cc[-1] - cc[0] != crop_size[1])):
                    continue

                # rotate and clip the current frame
                new_frame_clip = frames[i][:, :, None] * np.ones((1, 1, 3))
                rot_mat = cv2.getRotationMatrix2D((xc0, yc0), angles[i], 1)
                new_frame_clip = cv2.warpAffine(new_frame_clip.astype('float32'),
                                                rot_mat, crop_size).astype(frames.dtype)

                # overlay a circle on the mouse
                if i >= lpad and i <= pad + dur:
                    cv2.circle(new_frame_clip, (xc0, yc0), 3,
                               (255, 0, 0), -1)

                # superimpose the clipped mouse
                old_frame = crowd_movie[i]
                new_frame = np.zeros_like(old_frame)
                new_frame[rr[0]:rr[-1], cc[0]:cc[-1]] = new_frame_clip

                # zero out based on min_height before taking the non-zeros
                new_frame[new_frame < min_height] = 0
                old_frame[old_frame < min_height] = 0

                new_frame_nz = new_frame > 0
                old_frame_nz = old_frame > 0

                blend_coords = np.logical_and(new_frame_nz, old_frame_nz)
                overwrite_coords = np.logical_and(new_frame_nz, ~old_frame_nz)

                old_frame[blend_coords] = .5 * old_frame[blend_coords] \
                                          + .5 * new_frame[blend_coords]
                old_frame[overwrite_coords] = new_frame[overwrite_coords]

                crowd_movie[i] = old_frame

    return crowd_movie

def make_crowd_movie_grouped(state_idx,
                     num_taus,
                     posteriors,
                     pad=30,
                     raw_size=(512, 424),
                     crop_size=(80, 80),
                     offset=(50, 50),
                     scale=.5,
                     min_height=10,
                     **kwargs):
    '''
    Adapted from https://github.com/dattalab/moseq2-viz/blob/release/moseq2_viz/viz.py

    Creates crowd movie video numpy array.
    Parameters
    ----------
    dataset (list of dicts): list of dictionaries containing data
    slices (np.ndarray): video slices of specific syllable label
    pad (int): number of frame padding in video
    raw_size (tuple): video dimensions.
    frame_path (str): variable to access frames in h5 file
    crop_size (tuple): mouse crop size
    offset (tuple): centroid offsets from cropped videos
    scale (int): mouse size scaling factor.
    min_height (int): minimum max height from floor to use.
    kwargs (dict): extra keyword arguments
    Returns
    -------
    crowd_movie (np.ndarray): crowd movie for a specific syllable.
    '''
    slices = extract_syllable_slices(state_idx, posteriors)

    xc0, yc0 = crop_size[1] // 2, crop_size[0] // 2
    xc = np.arange(-xc0, xc0 + 1, dtype='int16')
    yc = np.arange(-yc0, yc0 + 1, dtype='int16')

    durs = []
    for these_slices in slices:
        for slc in these_slices:
            durs.append(slc.stop - slc.start)

    if len(durs) == 0:
        print("no valid syllables found for state", state_idx)
        return
    max_dur = np.max(durs)

    # Initialize the crowd movie
    crowd_movie = np.zeros((max_dur + pad * 2, raw_size[1], raw_size[0], 3),
                           dtype='uint8')

    for these_slices, posterior in zip(slices, posteriors):
        data = posterior.data
        for slc in these_slices:
            lpad = min(pad, slc.start)
            rpad = min(pad, len(data['frames']) - slc.stop)
            dur = slc.stop - slc.start
            padded_slc = slice(slc.start - lpad, slc.stop + rpad)
            centroid_x = data['centroid_x_px'][padded_slc] + offset[0]
            centroid_y = data['centroid_y_px'][padded_slc] + offset[1]
            angles = np.rad2deg(data['angles'][padded_slc])
            frames = (data['frames'][padded_slc] / scale).astype('uint8')
            flips = np.zeros(angles.shape, dtype='bool')

            for i in range(lpad + dur + rpad):
                if np.any(np.isnan([centroid_x[i], centroid_y[i]])):
                    continue

                rr = (yc + centroid_y[i]).astype('int16')
                cc = (xc + centroid_x[i]).astype('int16')

                if (np.any(rr < 1)
                        or np.any(cc < 1)
                        or np.any(rr >= raw_size[1])
                        or np.any(cc >= raw_size[0])
                        or (rr[-1] - rr[0] != crop_size[0])
                        or (cc[-1] - cc[0] != crop_size[1])):
                    continue

                # rotate and clip the current frame
                new_frame_clip = frames[i][:, :, None] * np.ones((1, 1, 3))
                rot_mat = cv2.getRotationMatrix2D((xc0, yc0), angles[i], 1)
                new_frame_clip = cv2.warpAffine(new_frame_clip.astype('float32'),
                                                rot_mat, crop_size).astype(frames.dtype)

                # overlay a circle on the mouse
                if i >= lpad and i <= pad + dur:
                    cv2.circle(new_frame_clip, (xc0, yc0), 3,
                               (255, 0, 0), -1)

                # superimpose the clipped mouse
                old_frame = crowd_movie[i]
                new_frame = np.zeros_like(old_frame)
                new_frame[rr[0]:rr[-1], cc[0]:cc[-1]] = new_frame_clip

                # zero out based on min_height before taking the non-zeros
                new_frame[new_frame < min_height] = 0
                old_frame[old_frame < min_height] = 0

                new_frame_nz = new_frame > 0
                old_frame_nz = old_frame > 0

                blend_coords = np.logical_and(new_frame_nz, old_frame_nz)
                overwrite_coords = np.logical_and(new_frame_nz, ~old_frame_nz)

                old_frame[blend_coords] = .5 * old_frame[blend_coords] \
                                          + .5 * new_frame[blend_coords]
                old_frame[overwrite_coords] = new_frame[overwrite_coords]

                crowd_movie[i] = old_frame

    return crowd_movie

def make_crowd_movie_indiv(state_idx,
                     posteriors,
                     pad=30,
                     raw_size=(512, 424),
                     crop_size=(80, 80),
                     offset=(50, 50),
                     scale=.5,
                     min_height=10,
                     **kwargs):
    '''
    Adapted from https://github.com/dattalab/moseq2-viz/blob/release/moseq2_viz/viz.py

    Creates crowd movie video numpy array.
    Parameters
    ----------
    dataset (list of dicts): list of dictionaries containing data
    slices (np.ndarray): video slices of specific syllable label
    pad (int): number of frame padding in video
    raw_size (tuple): video dimensions.
    frame_path (str): variable to access frames in h5 file
    crop_size (tuple): mouse crop size
    offset (tuple): centroid offsets from cropped videos
    scale (int): mouse size scaling factor.
    min_height (int): minimum max height from floor to use.
    kwargs (dict): extra keyword arguments
    Returns
    -------
    crowd_movie (np.ndarray): crowd movie for a specific syllable.
    '''
    slices = extract_syllable_slices_indiv(state_idx, posteriors)

    xc0, yc0 = crop_size[1] // 2, crop_size[0] // 2
    xc = np.arange(-xc0, xc0 + 1, dtype='int16')
    yc = np.arange(-yc0, yc0 + 1, dtype='int16')

    durs = []
    for these_slices in slices:
        for slc in these_slices:
            durs.append(slc.stop - slc.start)

    if len(durs) == 0:
        print("no valid syllables found for state", state_idx)
        return
    max_dur = np.max(durs)

    # Initialize the crowd movie
    crowd_movie = np.zeros((max_dur + pad * 2, raw_size[1], raw_size[0], 3),
                           dtype='uint8')

    for these_slices, posterior in zip(slices, posteriors):
        data = posterior.data
        for slc in these_slices:
            lpad = min(pad, slc.start)
            rpad = min(pad, len(data['frames']) - slc.stop)
            dur = slc.stop - slc.start
            padded_slc = slice(slc.start - lpad, slc.stop + rpad)
            centroid_x = data['centroid_x_px'][padded_slc] + offset[0]
            centroid_y = data['centroid_y_px'][padded_slc] + offset[1]
            angles = np.rad2deg(data['angles'][padded_slc])
            frames = (data['frames'][padded_slc] / scale).astype('uint8')
            flips = np.zeros(angles.shape, dtype='bool')

            for i in range(lpad + dur + rpad):
                if np.any(np.isnan([centroid_x[i], centroid_y[i]])):
                    continue

                rr = (yc + centroid_y[i]).astype('int16')
                cc = (xc + centroid_x[i]).astype('int16')

                if (np.any(rr < 1)
                        or np.any(cc < 1)
                        or np.any(rr >= raw_size[1])
                        or np.any(cc >= raw_size[0])
                        or (rr[-1] - rr[0] != crop_size[0])
                        or (cc[-1] - cc[0] != crop_size[1])):
                    continue

                # rotate and clip the current frame
                new_frame_clip = frames[i][:, :, None] * np.ones((1, 1, 3))
                rot_mat = cv2.getRotationMatrix2D((xc0, yc0), angles[i], 1)
                new_frame_clip = cv2.warpAffine(new_frame_clip.astype('float32'),
                                                rot_mat, crop_size).astype(frames.dtype)

                # overlay a circle on the mouse
                if i >= lpad and i <= pad + dur:
                    cv2.circle(new_frame_clip, (xc0, yc0), 3,
                               (255, 0, 0), -1)

                # superimpose the clipped mouse
                old_frame = crowd_movie[i]
                new_frame = np.zeros_like(old_frame)
                new_frame[rr[0]:rr[-1], cc[0]:cc[-1]] = new_frame_clip

                # zero out based on min_height before taking the non-zeros
                new_frame[new_frame < min_height] = 0
                old_frame[old_frame < min_height] = 0

                new_frame_nz = new_frame > 0
                old_frame_nz = old_frame > 0

                blend_coords = np.logical_and(new_frame_nz, old_frame_nz)
                overwrite_coords = np.logical_and(new_frame_nz, ~old_frame_nz)

                old_frame[blend_coords] = .5 * old_frame[blend_coords] \
                                          + .5 * new_frame[blend_coords]
                old_frame[overwrite_coords] = new_frame[overwrite_coords]

                crowd_movie[i] = old_frame

    return crowd_movie

# def make_indiv_movie(state, posteriors, pad=30):
#     raw_size = posteriors[0].data['frames'][0].shape
#     num_taus = len(posteriors[0].model.taus)
#
#     state_slices = [[] for tau in range(num_taus)]
#     state_frames = [[] for tau in range(num_taus)]
#     for posterior in posteriors:
#         data = posterior.data
#         model = posterior.model
#         num_discrete_states = model.num_discrete_states
#         for tau in range(num_taus):
#             state_idx = state*num_taus + tau
#             slices = extract_syllable_slices(state_idx, [posterior])
#             if len(slices[0]) == 0:
#                 print("no valid syllables found for state", state_idx)
#             elif len(slices[0]) != 0 and state_slices[tau] == []:
#                 state_slices[tau] = [slices[0][0]]
#                 slc = slices[0][0]
#                 lpad = min(pad, slc.start)
#                 rpad = min(pad, len(data['frames']) - slc.stop)
#                 dur = slc.stop - slc.start
#                 padded_slc = slice(slc.start - lpad, slc.stop + rpad)
#                 state_frames[tau] = (data['frames'][padded_slc]).astype('uint8')
#
#
#     durs = []
#     for slc in state_slices:
#         if len(slc) is not 0:
#             slc = slc[0]
#             durs.append(slc.stop - slc.start)
#
#     max_dur = np.max(durs)
#
#     # Initialize the crowd movie
#     crowd_movie = np.zeros((max_dur + pad * 2, raw_size[1]*num_taus, raw_size[0]),
#                            dtype='uint8')
#
#     for it, frame, slc in zip(range(num_taus), state_frames,state_slices):
#         slc = slc[0]
#         dur = slc.stop - slc.start
#         crowd_movie[:dur+pad*2,raw_size[1]*it:raw_size[1]*(it+1),:] = frame
#
#     crowd_movie = crowd_movie[:,:, :, None] * np.ones((1, 1, 3))
#
#     crowd_movie = cv2.normalize(crowd_movie, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
#     crowd_movie = crowd_movie.transpose([0, 3, 1, 2])
#     wandb.log(
#         {'temp': wandb.Video(crowd_movie, fps=30, format="mp4")})
#     return crowd_movie