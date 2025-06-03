from collections import defaultdict
import functools
import numpy as np
from matplotlib import pyplot as plt, colors as mcolors
import seaborn as sns
import h5py
from pathlib import Path

# arrays are NxM, (observations are rows, columns are time)


root = Path(__file__).resolve().parent.parent.resolve()

def gather(write=True):
    histories = dict()
    for dir in (root / "simulations/zero_sims_resized/" ).iterdir():
        model = dir.stem
        fl = dir / "sim_all_30s.h5"
        with h5py.File(fl) as fp:
            grps = list(fp)
            presence = [fp[grp]['history'][:].sum(axis=0) for grp in grps]
        histories[model] = np.vstack(presence)
        if write:
            write_to_file = dir / "hosp_presence.csv"
            np.savetxt(write_to_file, histories[model], delimiter=',')
            print(f"presence written to {write_to_file}")
    return histories

def load(fname="hosp_presence.csv"):
    histories = {
        dir.stem: np.loadtxt(dir/fname, delimiter=',')
        for dir in (root / "simulations/zero_sims_resized/" ).iterdir()
    }

    return histories


def compute_heatmaps(hists, write=True):
    hms = {model: heatmap(hist_vals)[1] for model, hist_vals in hists.items()}
    # adjust temporal for dt
    hms['temporal'] = np.hstack([
        (hms['temporal'][:,:-1:2] + hms['temporal'][:, 1::2]) / 2,
        hms['temporal'][:,-1:],
    ])

    if write:
        for model, hmarr in hms.items():
            write_to_file = root / "simulations/zero_sims_resized" / model / "hosp_presence_heat.csv"
            np.savetxt(write_to_file, hmarr, delimiter=',')
            print(f"heatmap written to {write_to_file}")

    return hms

def load_heatmaps():
    return load("hosp_presence_heat.csv")

_BASE_DATE = np.datetime64('2011-01', 'M')
_MAX_DATE = np.datetime64('2018-12', 'M')
def set_xdatelabels(hist, ax: plt.Axes, spacing=12, dt=1.0):
    # here, we want to get succinct date formats
    # we map the times onto the timedeltas of the simulations
    N = hist.shape[1]
    emp_max_date = (np.timedelta64(int(N*dt), 'D') + _BASE_DATE).astype('datetime64[M]')
    
    ax_max_date = np.min([emp_max_date, _MAX_DATE])

    dates = np.arange(_BASE_DATE, ax_max_date, np.timedelta64(spacing, 'M'))

    label_pos = (dates.astype('datetime64[D]').astype('int64') - _BASE_DATE.astype('datetime64[D]').astype('int64')) / dt

    ax.set_xticks(label_pos, dates.astype('datetime64[Y]'), rotation=45)

    return ax.get_xticks()

def heatmap(hist, nmax=30):
    # hist is 2D
    bins = np.arange(nmax+2)
    hist_heat = np.array([
        np.histogram(hist[:, i], bins=bins)[0]
        for i in range(hist.shape[1])
    ]).T

    return bins, hist_heat

@functools.wraps(heatmap)
def plot_heatmap(*args, **kwargs):
    bins, hist_heat = heatmap(*args, **kwargs)

    ax = sns.heatmap(hist_heat)

    return ax

_DTS = defaultdict(lambda: 1.0)
_DTS['temporal'] = 0.5
def single_heatmap(model, hist, bins=(540, 31), norm=None, ax=None, cbar=False):
    if ax is None:
        fig, ax = plt.subplots()
    
    if norm is None:
        norm = mcolors.Normalize()
    
    dt = _DTS[model]
    ts = (np.arange(hist.shape[1]) * dt).reshape((1, -1))
    ts_arr = np.repeat(ts, hist.shape[0], axis=0)

    *_, coll = ax.hist2d(
        ts_arr.flatten(), 
        30 - hist.flatten(),
        bins=bins,
        density=True,
        norm=norm,
    )

    if cbar:
        ax.figure.colorbar(coll, ax=ax)

    return coll, ax

_ax_order = {
    'temporal': 0,
    'snapshot': 1,
    'static': 2,
    'naive_static': 3,
}
def final_heatmap(hists, norm=mcolors.PowerNorm(0.5)):

    fig, axs = plt.subplots(nrows=4, figsize=[16, 9], sharex=True)
    axs_flat = axs.flatten()
    for k, h in hists.items():
        ax = axs_flat[_ax_order[k]]
        coll, _ = single_heatmap(k, h, norm=norm, ax=ax)
        norm = coll.norm
        ax.set_ylabel(k.replace('_', ' '))
    set_xdatelabels(h, axs_flat[-1])
    fig.subplots_adjust(hspace=0.3)
    fig.colorbar(coll, ax=axs, label='Density', extend='max')
    
    return fig, axs

def mega_heatmap(vmax=1500):
    hms = load_heatmaps() 
    fig, axs = plt.subplots(nrows=4, figsize=[16, 9], sharex=True)
    for ax, (k, hmv) in zip(axs.flatten(), hms.items()):
        sns.heatmap(data=hmv[::-1,:], vmax=vmax, ax=ax, cmap='inferno', cbar=False)
        ax.set_title(k)
        ax.set_ylim([-0.5, 30.5])
    set_xdatelabels(hmv, ax)
    fig.subplots_adjust(hspace=0.4)
    fig.colorbar(ax.collections[0], ax=axs, label='Count', extend='max')

    return fig

if __name__ == "__main__":
    gather()