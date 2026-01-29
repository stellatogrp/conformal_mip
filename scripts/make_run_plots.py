from __future__ import annotations

import argparse
import json
import os
import time

import mlflow
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Patch
from mlflow import MlflowClient
from test import get_most_recent_run_from_experiment_name

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "font.size": 12,
})


def grouped_colored_boxplot(
    data,                      # dict: {a: {b: c}}
    order_a=None,              # optional list specifying a order
    order_b=None,              # optional list specifying b order
    b_to_color=None,           # optional dict {b: color}
    gap_between_a=1.0,         # spacing between groups (a's)
    box_width=0.8,
    showfliers=False,
    xlabel="a",
    ylabel="c",
    title=None,
    ax=None,
    xline=None,
    ymin=None,
    scale="log",
    hide_xlabel=False,
):
    """
    data[a][b] = c, where c is:
      - a list/np.array of numbers, OR
      - a single number
    Makes a grouped boxplot: groups by 'a', colors by 'b'.
    """
    minimum_y_on_scale = ymin

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    else:
        fig = ax.figure

    # ---- gather keys & normalize c to list-like ----
    a_keys = list(data.keys()) if order_a is None else list(order_a)

    b_keys_set = set()
    for a in a_keys:
        if a in data:
            b_keys_set |= set(data[a].keys())
    b_keys = sorted(b_keys_set) if order_b is None else list(order_b)

    # default colors if not provided
    if b_to_color is None:
        cycle = plt.rcParams['axes.prop_cycle'].by_key().get('color', [])
        if not cycle:
            cycle = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"]
        b_to_color = {b: cycle[i % len(cycle)] for i, b in enumerate(b_keys)}

    # ---- build plotting arrays: one box per (a,b) that exists ----
    values = []
    positions = []
    facecolors = []
    labels = []

    n_b = len(b_keys)

    for i_a, a in enumerate(a_keys):
        if a not in data:
            continue
        base = i_a * (n_b + gap_between_a)
        for j_b, b in enumerate(b_keys):
            if b not in data[a]:
                continue
            c = data[a][b]
            if np.isscalar(c):
                c = [c]
            else:
                c = list(c)

            pos = base + (j_b - (n_b - 1) / 2.0)

            values.append(c)
            positions.append(pos)
            facecolors.append(b_to_color[b])
            labels.append((a, b))

    if xline is not None:
        ax.hlines(y=xline, xmin=positions[0]-1, xmax=positions[-1]+1, color="black", linestyle="--", linewidth=1.0, alpha=0.75)

    # ---- compute custom box stats (5–95 box) ----
    stats = []
    for c in values:
        c = np.asarray(c)
        c = np.maximum(c, 1e-6)
        stats.append(dict(
            q1=np.percentile(c, 5),
            med=np.percentile(c, 50),
            q3=np.percentile(c, 95),
            whislo=np.min(c),
            whishi=np.max(c),
            fliers=[],
        ))

    all_vals = []
    for c in values:
        all_vals.extend(c)

    all_vals = np.asarray(all_vals)

    # ---- y-scale and safe limits (never clip boxes) ----
    ymin = np.min(all_vals)
    ymax = np.max(all_vals)

    # handle nonnegative data nicely
    if ymin >= 0:
        ymin = 0.0

    # add headroom so whiskers never touch the top
    pad = 0.05 * (ymax - ymin if ymax > ymin else max(1.0, ymax))
    ymax += pad

    bp = ax.bxp(
        stats,
        positions=positions,
        widths=box_width,
        patch_artist=True,
        showfliers=showfliers,
        manage_ticks=False,
    )

    for patch, fc in zip(bp["boxes"], facecolors):
        patch.set_facecolor(fc)
        patch.set_alpha(0.75)
    box_edge_lw = 0.6  # NEW: thinner
    for patch, fc in zip(bp["boxes"], facecolors):
        patch.set_facecolor(fc)
        patch.set_alpha(0.75)
        patch.set_linewidth(box_edge_lw)
        patch.set_edgecolor("black")

    for med in bp["medians"]:
        med.set_color("black")
        med.set_linewidth(1.5)

    # ---- x ticks at group centers (one tick per a) ----
    group_centers = []
    group_labels = []
    for i_a, a in enumerate(a_keys):
        if a not in data:
            continue
        base = i_a * (n_b + gap_between_a)
        group_centers.append(base)
        group_labels.append(str(a) if not hide_xlabel else "")

    ax.set_xticks(group_centers)
    ax.set_xticklabels(group_labels, rotation=0)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)

    # ---- legend for b colors ----
    handles = [Patch(facecolor=b_to_color[b], edgecolor="black", alpha=0.75, label=str(b))
               for b in b_keys]

    mid_left = (n_b - 1) // 2
    mid_right = mid_left + 1 if n_b % 2 == 0 else mid_left

    handles = []
    for j_b, b in enumerate(b_keys):
        h = Patch(
            facecolor=b_to_color[b],
            edgecolor="black",
            alpha=0.75,
            label=str(b),
            hatch=None
        )
        handles.append(h)

    fig.legend(
        handles=handles,
        title="Method",
        loc="lower center",
        ncol=max(1, len(b_keys)//2),
        fontsize="small",
        frameon=False,
    )

    fig.subplots_adjust(bottom=0.23)

    ax.margins(x=0.02)
    ax.grid(axis="y", alpha=0.25)

    if scale is not None:
        if scale == "symlog":
            ax.set_yscale(scale, linthresh=1e-3)
        else:
            ax.set_yscale(scale)
    ax.set_ylim(bottom=minimum_y_on_scale, top=ymax * 1.2)

    # ---- vertical dotted separators between x-groups (a) ----
    if len(group_centers) > 1:
        centers = np.asarray(group_centers, dtype=float)
        separators = (centers[:-1] + centers[1:]) / 2.0

        for x in separators:
            ax.axvline(
                x,
                linestyle=":",
                linewidth=1.0,
                color="0.6",
                alpha=0.8,
                zorder=0.4,   # below boxes, above grid
            )

    # keep plot elements above the shading
    ax.set_axisbelow(False)
    return fig, ax



def plot(
    data_csv_path: str,
    epsilon: float,
    experiment_name: str
):
    if not os.path.exists('outputs/plots'):
        os.makedirs('outputs/plots')

    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    names_in_exp = ['cp', 'solver']
    with open(data_csv_path) as f:
            results = json.load(f)
    cols = results['columns']
    data = results['data']
    data = {col: [d[i] for d in data] for i, col in enumerate(cols)}

    times = [data[name + '_time'] for name in names_in_exp]
    maxtime = max([max(t) for t in times])
    mintime = min([min(t) for t in times])

    for name in names_in_exp:
        rel_subopts = data[name + '_rel_subopt']
        times = data[name + '_time']
        times_order = np.argsort(times)
        times = np.array(times)[times_order]
        rel_subopts = np.array(rel_subopts)[times_order]
        correct = [c < epsilon and t < 599 for c, t in zip(rel_subopts, times)]
        cumsumcorrect = np.cumsum(correct) / len(correct) * 100
        cumsumcorrect = np.append(cumsumcorrect, cumsumcorrect[-1])
        times = np.append(times, 600)
        ax.set_xlim([mintime, maxtime])
        ax.set_xticklabels([0, 10, 100, maxtime])
        ax.set_title('solved/time for ' + experiment_name)
        ax.set_xscale("log")
        ax.plot(times, cumsumcorrect, label=name, color='blue', linestyle='-' if name=='cp' else '--')
        ax.grid(True) # Adds gridlines

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles, labels=labels, fontsize='small', ncol=4)
    ax.set_ylabel(f"Percent solved within {epsilon*100}\% tolerance (\%)")
    ax.set_xlabel("Time (s)")
    plt.tight_layout()
    plt.savefig(f'outputs/plots/solved_plot_{experiment_name}_{time.time()}.pdf')
    plt.close()

    names_in_exp = names_in_exp + ['solver1', 'solver3']
    rel_subs = {experiment_name: {name: data[name + '_rel_subopt'] for name in names_in_exp}}
    all_times = {experiment_name: {name: data[name + '_time'] for name in names_in_exp}}
    all_nodes = {experiment_name: {name: data[name + '_n_nodes'] for name in names_in_exp}}

    fig, ax = plt.subplots(3, 1, figsize=(6, 7))
    grouped_colored_boxplot(rel_subs,
        ax=ax[0], xlabel="", ylabel="Suboptimality", scale="log", xline=epsilon,
        ymin=1e-6, hide_xlabel=True
    )
    grouped_colored_boxplot(
        all_times, ax=ax[1], xlabel="", ylabel="Solve time (s)",  scale="log",
        hide_xlabel=True
    )
    grouped_colored_boxplot(
        all_nodes, ax=ax[2], xlabel="Experiment", ylabel="Number of nodes", scale="log",
        ymin=1,
    )
    plt.tight_layout()
    fig.subplots_adjust(bottom=0.18)
    plt.savefig(f'outputs/plots/solved_boxplots_{experiment_name}_{time.time()}.pdf')
    plt.close()


def get_result_csv_from_mlflow_name(mlflow_name: str, filestart='test_averages'):
    if mlflow_name is None:
        return None
    # set mlflow tracking uri
    tracking_uri = "mlruns"
    mlflow.set_tracking_uri(tracking_uri)
    _ = MlflowClient()
    # retrieve mlflow run
    run_id = mlflow.search_runs(
        filter_string=f"attributes.run_name='{mlflow_name}'",
        search_all_experiments=True
    )['run_id'].iloc[0]
    mlrun = mlflow.get_run(run_id=run_id)
    artefact_dir = mlrun.info.artifact_uri 
    results = [f for f in os.listdir(artefact_dir) if f.startswith(filestart)]
    if len(results) == 0:
        return None
    results = sorted(results)
    file = results[-1]
    return artefact_dir + '/' + file


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", type=str, default="CFLP-toy")
    parser.add_argument("--solver", type=str, default="gurobi")
    args = parser.parse_args()

    mlrun = get_most_recent_run_from_experiment_name(args.problem, args.solver)
    run_name = mlrun.data.tags["mlflow.runName"]
    data_csv_path = get_result_csv_from_mlflow_name(run_name, 'test_results')

    plot(data_csv_path, epsilon=0.001, experiment_name=args.problem)
