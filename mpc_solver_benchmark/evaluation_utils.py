import os
from typing import List
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from typing import List, Optional, Tuple
from acados_template import latexify_plot

from .simulation_results import SimulationResultsClosedLoop, SimulationResultsBase, SimulationResultsOpenLoop

FIGURE_DIR = "figures"

latexify_plot()

def plot_bar_plot_helper(plot_data: List[Tuple[float, str]], labels, ylabel, title: str = '', width: float=0.8, figure_filename: Optional[str]=None, figsize=None):

    fig, ax = plt.subplots(figsize=figsize)
    num_entries = len(labels)
    bottom = np.zeros(num_entries)

    for timing, label in plot_data:
        if not np.all(timing == 0):
            plt.bar(labels, timing, width, label=label, bottom=bottom)
            bottom += timing

    ax.set_title(title)
    plt.xticks(rotation=10)
    plt.grid(axis="y")
    plt.ylabel(ylabel)
    plt.tight_layout()
    ax.legend()

    if figure_filename is not None:
        figure_filename = os.path.join(FIGURE_DIR, figure_filename)
        plt.savefig(figure_filename)
        print(f"Saved figure to {figure_filename}")


def get_timings_data(results: List[SimulationResultsBase], labels: List[str], metric: str, per_iteration=False, timings_unit="us"):
    num_entries = len(labels)
    if num_entries != len(results):
        raise ValueError("Number of labels and result files do not match")

    closed_loop = isinstance(results[0], SimulationResultsClosedLoop)

    # setup combined dict
    combined_dict = {}
    for k in results[0].__dict__.keys():
        combined_dict[k] = []
        for result in results:
            combined_dict[k].append(getattr(result, k))

    if timings_unit == "ms":
        factor = 1e-3
    elif timings_unit == "us":
        factor = 1
    else:
        raise ValueError(f"Timings unit {timings_unit} not supported")

    time_tot = factor * np.array(combined_dict["time_preparation"]) + factor * np.array(combined_dict["time_feedback"])
    time_reg = factor * np.array(combined_dict["time_reg"])
    time_sim = factor * np.array(combined_dict["time_sim"])
    time_lin = factor * np.array(combined_dict["time_lin"])
    time_qp_solver_call = factor * np.array(combined_dict["time_qp_solver_call"])
    time_qp = factor * np.array(combined_dict["time_qp"])
    time_preparation = factor * np.array(combined_dict["time_preparation"])
    time_feedback = factor * np.array(combined_dict["time_feedback"])

    if closed_loop:
        if metric == "mean":
            metric_fun = np.mean
        elif metric == "min":
            metric_fun = np.min
        elif metric == "max":
            metric_fun = np.max
        elif metric == "median":
            metric_fun = np.median

        time_remaining = metric_fun(time_tot - time_lin - time_qp - time_reg, axis=1)
        time_tot = metric_fun(time_tot, axis=1)
        time_reg = metric_fun(time_reg, axis=1)
        time_sim = metric_fun(time_sim, axis=1)
        time_lin = metric_fun(time_lin, axis=1)
        time_qp = metric_fun(time_qp, axis=1)
        time_qp_solver_call = metric_fun(time_qp_solver_call, axis=1)
        time_preparation = metric_fun(time_preparation, axis=1)
        time_feedback = metric_fun(time_feedback, axis=1)
    else:
        if per_iteration:
            time_tot = time_tot / combined_dict["nlp_iter"]
            time_reg = time_reg / combined_dict["nlp_iter"]
            time_sim = time_sim / combined_dict["nlp_iter"]
            time_lin = time_lin / combined_dict["nlp_iter"]
            time_qp = time_qp / combined_dict["nlp_iter"]
            time_qp_solver_call = time_qp_solver_call / combined_dict["nlp_iter"]
            time_preparation = time_preparation / combined_dict["nlp_iter"]
            time_feedback = time_feedback / combined_dict["nlp_iter"]

    time_qp_remaining = time_qp - time_qp_solver_call
    time_remaining_lin = time_lin - time_sim
    # time_remaining = time_tot - time_lin - time_qp - time_reg

    return [(time_sim, "integrator"),
            (time_remaining_lin, "remaining linearization"),
            (time_qp_solver_call, "QP solver"),
            (time_qp_remaining, "QP preprocessing"),
            (time_reg, "regularization"),
            (time_remaining, "remaining"),
            (time_preparation, "preparation"),
            (time_feedback, "feedback"),
            (time_tot, "total")
    ]


def metric_to_tex_string(metric: str) -> str:
    if metric == "max":
        return "maximum"
    return metric

def plot_acados_timings_submodules(results: List[SimulationResultsBase], labels: List[str], n_runs, figure_filename: Optional[str]=None, metric="mean", per_iteration=False, timings_unit="ms"):

    closed_loop = isinstance(results[0], SimulationResultsClosedLoop)

    title = f"Time taken for each solver (runs: {n_runs})"
    if per_iteration:
        ylabel = "Time per NLP solver iteration"
    elif closed_loop:
        ylabel = f"{metric_to_tex_string(metric)} time per closed-loop iter. [{timings_unit}]"


    data = get_timings_data(results, labels, metric=metric, per_iteration=per_iteration, timings_unit=timings_unit)
    data = [d for d in data if not np.all(d[0] == 0)]
    data = [d for d in data if d[1] not in ["total", "preparation", "feedback"]]
    plot_bar_plot_helper(data, labels, ylabel, title, figure_filename=figure_filename)


def get_relative_suboptimality(results: List[SimulationResultsClosedLoop]):
    min_cost = np.min([result.cost_traj[-1] for result in results])
    return [1e2 * (result.cost_traj[-1]-min_cost) / min_cost for result in results]


def plot_acados_timings_real_time_split(results: List[SimulationResultsBase], labels: List[str], n_runs, figure_filename: Optional[str]=None, metric="mean", per_iteration=False, timings_unit="ms", fig_filename=None, title=None,
                                        figsize=None):

    closed_loop = isinstance(results[0], SimulationResultsClosedLoop)

    title_ = f"Time taken for each solver (runs: {n_runs})"
    if per_iteration:
        ylabel = "Time per NLP solver iteration"
    elif closed_loop:
        ylabel = f"{metric_to_tex_string(metric)} time per closed-loop iter. [{timings_unit}]"

    data = get_timings_data(results, labels, metric=metric, per_iteration=per_iteration, timings_unit=timings_unit)
    data = [d for d in data if not np.all(d[0] == 0)]
    data = [d for d in data if d[1] in ["preparation", "feedback"]]
    plot_bar_plot_helper(data, labels, ylabel, title_, figure_filename=figure_filename, figsize=figsize)

    if title is not None:
        plt.title(title)

    if fig_filename is not None:
        plt.savefig(fig_filename, bbox_inches="tight", transparent=True, pad_inches=0.05)
        print(f"\nstored figure in {fig_filename}")


def print_acados_timings_table_submodules(results: List[SimulationResultsBase], labels: List[str],
                               style: str ="markdown", metric="mean", per_iteration=False, timings_unit="ms"):
    if style not in ["latex", "markdown"]:
        raise ValueError(f"Style {style} not supported")

    data = get_timings_data(results, labels, metric=metric, per_iteration=per_iteration, timings_unit=timings_unit)

    print_time_reg = any([sum(getattr(r, "time_reg")) != 0 for r in results])
    column_names = ["Variant"] + [d[1] for d in data]

    if not print_time_reg:
        data = [d for d in data if d[1] != "regularization"]
    n_columns = len(column_names)

    table_string = ""
    if style == "latex":
        table_string += "\\begin{table}\n"
        table_string += "\\caption{" f"Timings in {timings_unit}" + "}\n"
        table_string += "\\begin{tabular}{lcccccc}\n"
        table_string += "\\toprule\n"
        table_string += " & ".join(column_names) + "\\\\\n"
    if style == "markdown":
        table_string = "| " + " | ".join(column_names) + " | \n"
        table_string += n_columns * "|---------" + "|\n"
    label_string_len = max([len(label) for label in labels]) + 2
    for i, label in enumerate(labels):
        if style == "markdown":
            table_string += f"| {label:{label_string_len}} | "
            for d in data:
                table_string += f"{d[0][i]:.0f} | "
            table_string += "\n"
        elif style == "latex":
            table_string += f"{label:{label_string_len}} "
            for d in data:
                table_string += f" & {d[0][i]:.0f}"
            table_string += "\\\\\n"

    if style == "latex":
        table_string += "\\bottomrule\n"
        table_string += "\\end{tabular}\n"
        table_string += "\\end{table}\n"

    print(table_string)


def print_acados_timings_table_real_time_split(results: List[SimulationResultsBase], labels: List[str],
                               style: str ="markdown", metric="mean", per_iteration=False, timings_unit="ms",
                               include_relative_suboptimality=False):
    if style not in ["latex", "markdown"]:
        raise ValueError(f"Style {style} not supported")

    data = get_timings_data(results, labels, metric=metric, per_iteration=per_iteration, timings_unit=timings_unit)
    data = [d for d in data if d[1] in ["preparation", "feedback", "total"]]
    # data += [([max(r.qp_iter) for r in results], 'QP iter')]
    if include_relative_suboptimality:
        data += [(get_relative_suboptimality(results), 'rel. subopt. \%')]

    column_names = ["Variant"] + [d[1] for d in data]
    n_columns = len(column_names)

    timing_formatter = ".2f"

    table_string = ""
    if style == "latex":
        table_string += "\\begin{table}\n"
        table_string += "\\caption{" f"Timings in {timings_unit}" + "}\n"
        table_string += "\\begin{tabular}{lcccccc}\n"
        table_string += "\\toprule\n"
        table_string += " & ".join(column_names) + "\\\\ \\midrule\n"
    if style == "markdown":
        table_string = "| " + " | ".join(column_names) + " | \n"
        table_string += n_columns * "|---------" + "|\n"
    label_string_len = max([len(label) for label in labels]) + 2
    for i, label in enumerate(labels):
        if style == "markdown":
            table_string += f"| {label:{label_string_len}} | "
            for d in data:
                table_string += f"{d[0][i]:{timing_formatter}} | "
            table_string += "\n"
        elif style == "latex":
            table_string += f"{label:{label_string_len}} "
            for d in data:
                table_string += f" & {d[0][i]:{timing_formatter}}"
            table_string += "\\\\\n"

    if style == "latex":
        table_string += "\\bottomrule\n"
        table_string += "\\end{tabular}\n"
        table_string += "\\end{table}\n"

    print(table_string)
    print("")


def print_closed_loop_costs_timings_table(results: List[SimulationResultsClosedLoop], labels: List[str], style: str ="markdown"):
    if style not in ["latex", "markdown"]:
        raise ValueError(f"Style {style} not supported")
    column_names = ["Variant", "rel. suboptimality \%", "max computation time"]
    min_cost = np.min([result.cost_traj[-1] for result in results])
    n_columns = len(column_names)
    label_string_len = max([len(label) for label in labels]) + 2

    if style == "latex":
        table_string = "\\begin{table}\n"
        table_string += "\\begin{tabular}{lcccccc}\n"
        table_string += "\\toprule\n"
        table_string += " & ".join(column_names) + "\\\\\\midrule\n"

        for label, result in zip(labels, results):
            rel_subopt = 1e2 * (result.cost_traj[-1]-min_cost) / min_cost
            table_string += f"{label:{label_string_len}} & {rel_subopt:.2f} & {np.max(result.time_tot):.0f} \\\\\n"
        table_string += "\\bottomrule\n"
        table_string += "\\end{tabular}\n"
        table_string += "\\end{table}\n"


    elif style == "markdown":
        table_string = "| " + " | ".join(column_names) + " | \n"
        table_string += n_columns * "|---------" + "|\n"

        for label, result in zip(labels, results):
            rel_subopt = 1e2 * (result.cost_traj[-1]-min_cost) / min_cost
            table_string += f"| {label:{label_string_len}} | {rel_subopt:.2f} | {np.max(result.time_tot):.0f} |\n"

    print(table_string)
    print("")



def plot_trajectories(
    results: List[SimulationResultsBase],
    labels_list,
    x_labels_list=None,
    u_labels_list=None,
    idxbu=[],
    lbu=None,
    ubu=None,
    X_ref=None,
    U_ref=None,
    fig_filename=None,
    x_min=None,
    x_max=None,
    title=None,
    idxpx=None,
    idxpu=None,
    color_list=None,
    linestyle_list=None,
    single_column = False,
    alpha_list = None,
    time_label = None,
    idx_xlogy = None,
    show_legend = True,
    bbox_to_anchor = None,
    ncol_legend = 2,
    figsize=None,
):
    if isinstance(results[0].x_traj, list):
        nx = results[0].x_traj[0].shape[0]
        nu = results[0].u_traj[0].shape[0]
    else:
        nx = results[0].x_traj.shape[1]
        nu = results[0].u_traj.shape[1]
    Ntraj = len(results)

    if idxpx is None:
        idxpx = list(range(nx))
    if idxpu is None:
        idxpu = list(range(nu))

    if color_list is None:
        color_list = [f"C{i}" for i in range(Ntraj)]
    if linestyle_list is None:
        linestyle_list = Ntraj * ['-']
    if alpha_list is None:
        alpha_list = Ntraj * [0.8]

    if idx_xlogy is None:
        idx_xlogy = []

    if time_label is None:
        time_label = "$t$"

    if x_labels_list is None:
        x_labels_list = [f"$x_{i}$" for i in range(nx)]
    if u_labels_list is None:
        u_labels_list = [f"$u_{i}$" for i in range(nu)]

    nxpx = len(idxpx)
    nxpu = len(idxpu)
    nrows = max(nxpx, nxpu)

    if figsize is None:
        if single_column:
            figsize = (6.0, 2*(nxpx+nxpu+1))
        else:
            figsize = (10, (nxpx+nxpu))

    if single_column:
        fig, axes = plt.subplots(ncols=1, nrows=nxpx+nxpu, figsize=figsize, sharex=True)
    else:
        fig, axes = plt.subplots(ncols=2, nrows=nrows, figsize=figsize, sharex=True)
        axes = np.ravel(axes, order='F')

    if title is not None:
        axes[0].set_title(title)

    for i in idxpx:
        isubplot = idxpx.index(i)
        for result, label, color, linestyle, alpha in zip(results, labels_list, color_list, linestyle_list, alpha_list):

            if isinstance(result.x_traj, list):
                vals = [x[i] for x in result.x_traj]
            else:
                vals = result.x_traj[:, i]
            axes[isubplot].plot(result.t_traj, vals, label=label, alpha=alpha, color=color, linestyle=linestyle)

        if X_ref is not None:
            axes[isubplot].step(
                result.t_traj,
                X_ref[:, i],
                alpha=0.8,
                where="post",
                label="reference",
                linestyle="dotted",
                color="k",
            )
        axes[isubplot].set_ylabel(x_labels_list[i])
        axes[isubplot].grid()
        axes[isubplot].set_xlim(result.t_traj[0], result.t_traj[-1])

        if i in idx_xlogy:
            axes[isubplot].set_yscale('log')

        if x_min is not None:
            axes[isubplot].set_ylim(bottom=x_min[i])

        if x_max is not None:
            axes[isubplot].set_ylim(top=x_max[i])

    for i in idxpu:
        for result, label, color, linestyle, alpha in zip(results, labels_list, color_list, linestyle_list, alpha_list):
            if isinstance(result.u_traj, list):
                vals = [u[i] for u in result.u_traj]
            else:
                vals = result.u_traj[:, i]
            axes[i+nrows].step(result.t_traj, np.append([vals[0]], vals), label=label, alpha=alpha, color=color, linestyle=linestyle)

        if U_ref is not None:
            axes[i+nrows].step(result.t_traj, np.append([U_ref[0, i]], U_ref[:, i]), alpha=0.8,
                               label="reference", linestyle="dotted", color="k")

        axes[i+nrows].set_ylabel(u_labels_list[i])
        axes[i+nrows].grid()

        if i in idxbu:
            axes[i+nrows].hlines(
                ubu[i], result.t_traj[0], result.t_traj[-1], linestyles="dashed", alpha=0.4, color="k"
            )
            axes[i+nrows].hlines(
                lbu[i], result.t_traj[0], result.t_traj[-1], linestyles="dashed", alpha=0.4, color="k"
            )
            axes[i+nrows].set_xlim(result.t_traj[0], result.t_traj[-1])
            bound_margin = 0.05
            u_lower = (1-bound_margin) * lbu[i] if lbu[i] > 0 else (1+bound_margin) * lbu[i]
            axes[i+nrows].set_ylim(bottom=u_lower, top=(1+bound_margin) * ubu[i])

    axes[nxpx+nxpu-1].set_xlabel(time_label)
    if not single_column:
        axes[nxpx-1].set_xlabel(time_label)

    if bbox_to_anchor is None and single_column:
        bbox_to_anchor=(0.5, -0.75)
    elif bbox_to_anchor is None:
        bbox_to_anchor=(0.5, -1.5)

    if show_legend:
        axes[nxpx+nxpu-1].legend(loc="lower center", ncol=ncol_legend, bbox_to_anchor=bbox_to_anchor)

    fig.align_ylabels()
    # fig.tight_layout()

    if not single_column:
        for i in range(nxpu, nxpx):
            fig.delaxes(axes[i+nrows])

    if fig_filename is not None:
        plt.savefig(fig_filename, bbox_inches="tight", transparent=True, pad_inches=0.05)
        print(f"\nstored figure in {fig_filename}")


def plot_pareto(
    points,
    colors,
    alphas,
    markers,
    fig_filename: Optional[str] = None,
    xlabel=None,
    ylabel=None,
    marker_legend: Optional[dict] = None,
    color_legend: Optional[dict] = None,
    alpha_legend: Optional[dict] = None,
    markersize_legend: Optional[dict] = None,
    markersizes: Optional[list] = None,
    title=None,
    ncol_legend=None,
    figsize=None,
    bbox_to_anchor=None,
    xscale="symlog",
    xlim=None,
):
    if figsize is None:
        figsize = (6.6, 4.5)
    fig, axes = plt.subplots(ncols=1, nrows=1, figsize=figsize)

    if markersizes is None:
        markersizes = len(points) * [9]

    for p, marker, color, alpha, markersize in zip(points, markers, colors, alphas, markersizes):
        axes.plot(p[0], p[1], color=color, marker=marker, alpha=alpha, markersize=markersize)

    legend_elements = []
    if color_legend is not None:
        legend_elements += [
            Line2D([0], [0], color=color, lw=4, label=f"{key}")
            for key, color in color_legend.items()
        ]
    if marker_legend is not None:
        legend_elements += [
            Line2D([0], [0], marker=marker, lw=0, color="k", label=f"{key}")
            for key, marker in marker_legend.items()
        ]
    if alpha_legend is not None:
        legend_elements += [
            Line2D([0], [0], marker="o", color="k", alpha=alpha, lw=0, label=f"{key}")
            for key, alpha in alpha_legend.items()
        ]
    if markersize_legend is not None:
        legend_elements += [
            Line2D([0], [0], marker="o", color="k", markersize=markersize, lw=0, label=f"{key}")
            for key, markersize in markersize_legend.items()
        ]

    if xlabel is not None:
        axes.set_xlabel(xlabel)
    if ylabel is not None:
        axes.set_ylabel(ylabel)

    if xscale == "log":
        axes.set_xscale("log")
    elif xscale == "symlog":
        axes.set_xscale("symlog", linthresh=0.1, subs=[2, 3, 4, 5, 6, 7, 8, 9])
    else:
        raise NotImplementedError(f"xscale = {xscale}")
    # axes.set_yscale("log")

    plt.grid()
    if ncol_legend is None:
        ncol_legend = 1
    if len(legend_elements) > 0:
        plt.legend(handles=legend_elements, ncol=ncol_legend, bbox_to_anchor=bbox_to_anchor)

    if xlim is not None:
        axes.set_xlim(xlim)

    if title is not None:
        axes.set_title(title)

    plt.tight_layout()
    if fig_filename is not None:
        fig_filename = os.path.join(os.getcwd(), fig_filename)
        plt.savefig(fig_filename, bbox_inches="tight", transparent=True, pad_inches=0.05)
        print(f"\nstored figure in {fig_filename}")


def plot_trajectories_xy_space(
    position_trajectories: list[np.ndarray],
    labels: list[str],
    circular_obstacles: Optional[list[np.ndarray]] = None,
    fig_filename = None,
    show_legend = True,
    bbox_to_anchor = None,
    ncol_legend = 2,
):
    # create figure
    fig, ax = plt.subplots(figsize=(6.6, 4.5))
    ax.axis("equal")

    for i, (position_trajectory, label) in enumerate(zip(position_trajectories, labels)):
        ax.plot(position_trajectory[:, 0], position_trajectory[:, 1], label=label, marker="o", markersize=2)

    if bbox_to_anchor is None:
        bbox_to_anchor=(0.5, -0.75)

    if show_legend:
        fig.legend(loc="lower center", ncol=ncol_legend, bbox_to_anchor=bbox_to_anchor)

    if fig_filename is not None:
        plt.savefig(fig_filename, bbox_inches="tight", transparent=True, pad_inches=0.05)
        print(f"\nstored figure in {fig_filename}")

    # plot circular obstacles
    if circular_obstacles is None:
        circular_obstacles = []
    for obstacle in circular_obstacles:
        circle = plt.Circle(obstacle[:2], obstacle[2], color="r", fill=False)
        ax.add_artist(circle)

    ax.grid()
    ax.legend()
    ax.set_xlabel("x")
    ax.set_ylabel("y")


def plot_contraction_rate(results: List[SimulationResultsOpenLoop], labels_list: List[str], y_scale_log: bool = False):

    fig, axes = plt.subplots(nrows=1, ncols=1)

    for result, l in zip(results, labels_list):
        rate_estimates = result.primal_step_norm_traj[1:]/result.primal_step_norm_traj[:-1]
        axes.plot(rate_estimates, label=l)

    axes.grid()
    axes.set_xlabel('iteration $k$')
    axes.set_ylabel('empirical contraction rate $\hat{\kappa}$')
    axes.set_xlim(left = 0)
    axes.set_ylim(top = 1.25, bottom=-0.05)
    axes.legend()

    if y_scale_log:
        axes.set_yscale('log')

def plot_simplest_pareto(
    points,
    labels,
    colors,
    markers,
    fig_filename: Optional[str] = None,
    xlabel=None,
    ylabel=None,
    title=None,
    xlim=None,
    figsize=None,
    bbox_to_anchor=None,
    ncol_legend=None,
    xscale="log"
):
    latexify_plot()
    if figsize is None:
        figsize = (6.6, 4.5)
    fig, axes = plt.subplots(ncols=1, nrows=1, figsize=figsize)

    if xlabel is not None:
        axes.set_xlabel(xlabel)
    if ylabel is not None:
        axes.set_ylabel(ylabel)
    axes.set_xscale(xscale)

    plt.grid()
    plt.tight_layout()

    if title is not None:
        axes.set_title(title)

    assert len(points) == len(labels)
    assert len(colors) >= len(points)
    assert len(markers) >= len(points)

    legend_elements = []

    for p, label, marker, color in zip(points, labels, markers, colors):
        alpha = 1.0
        fillstyle = "full"
        color = color
        axes.plot(
            p[0],
            p[1],
            color=color,
            marker=marker,
            alpha=alpha,
            markersize=10,
            markeredgewidth=2,
            fillstyle=fillstyle,
        )
        legend_elements += [
            Line2D(
                [0],
                [0],
                marker=marker,
                alpha=alpha,
                markersize=10,
                markeredgewidth=2,
                fillstyle=fillstyle,
                color=color,
                lw=0,
                label=label,
            )
        ]

    if xlim is not None:
        axes.set_xlim(xlim)

    if ncol_legend is None:
        ncol_legend = 1
    plt.legend(handles=legend_elements, ncol=ncol_legend, bbox_to_anchor=bbox_to_anchor)
    plt.tight_layout()
    if fig_filename is not None:
        plt.savefig(
            fig_filename, bbox_inches="tight", transparent=True, pad_inches=0.05
        )
        print(f"\nstored figure in {fig_filename}")
