import numpy as np
from typing import Tuple

from mpc_solver_benchmark.problems import formulate_cstr_ocp as formulate_ocp
from mpc_solver_benchmark.problems import CstrOcpOptions as OcpOptions
from mpc_solver_benchmark import (
    closed_loop_experiment,
    plot_acados_timings_submodules,
    get_results_filename,
    dataclass_to_string,
    AcadosIntegrator,
    AcadosSolverTimeVarying,
    AcadosSolverConstantReference,
    AcadosSolverTimeVaryingReference,
    print_closed_loop_costs_timings_table,
    get_results_from_filenames,
    plot_trajectories,
    plot_pareto,
    get_varying_fields,
    Reference,
)

PROBLEM_NAME = "cstr"
X0 = np.array([4.39000e-02, 2.43375e02, 3.29500e-01])

def setup_reference(dt_plant: float, n_sim: int) -> Tuple[Reference]:
    t_jump_1 = int(n_sim / 3) * dt_plant
    t_jump_2 = 2 * int(n_sim / 3) * dt_plant
    # steady-state
    xs = np.array([[0.878, 324.5, 0.659]]).T
    us = np.array([[300, 0.1]]).T
    # reference jump
    xs2 = np.array([0.7, 337, 0.75])
    us2 = np.array([305, 0.1])

    reference = Reference([t_jump_1, t_jump_2],
                          [np.concatenate((xs, us)),
                           np.concatenate((xs2, us2)),
                           np.concatenate((xs, us))])
    terminal_ref = reference.get_sub_reference(np.arange(xs.shape[0]))

    return reference, terminal_ref


def create_options_list() -> list[OcpOptions]:
    options_list = []
    n_fine = 50
    n_coarse = 8
    # options_list.append(OcpOptions(N_horizon=(45, 5), sim_method_num_steps=1, known_reference=True, cost_discretization="INTEGRATOR", cl_costing=True))
    options_list.append(OcpOptions(N_horizon=n_coarse, sim_method_num_steps=2, known_reference=True, cost_discretization="INTEGRATOR", clc_horizon=4))
    options_list.append(OcpOptions(N_horizon=n_coarse, sim_method_num_steps=2, known_reference=True, cost_discretization="INTEGRATOR"))
    # options_list.append(OcpOptions(N_horizon=16, sim_method_num_steps=1, known_reference=True, cost_discretization="INTEGRATOR"))
    # options_list.append(OcpOptions(N_horizon=n_coarse, sim_method_num_steps=2, known_reference=True, cost_discretization="EULER"))
    # options_list.append(OcpOptions(N_horizon=n_fine, sim_method_num_steps=1, known_reference=True))
    # options_list.append(OcpOptions(N_horizon=n_fine, sim_method_num_steps=1, known_reference=True, cost_discretization="EULER"))
    return options_list


def setup_solver_from_opts(ocp_options: OcpOptions, reference: Reference, terminal_reference: Reference):
    ocp = formulate_ocp(ocp_options, reference=reference)

    if not ocp_options.known_reference:
        solver = AcadosSolverConstantReference(ocp, reference=reference, reference_terminal=terminal_reference)
    elif ocp_options.cost_discretization == "INTEGRATOR":
        solver = AcadosSolverTimeVarying(ocp)
    else:
        solver = AcadosSolverTimeVaryingReference(ocp, reference=reference, reference_terminal=terminal_reference)

    return solver


def get_labels(options_list: list[OcpOptions]):
    labels = []
    for opts in options_list:
        label = f"$N={opts.N_horizon}$ {'known reference' if opts.known_reference else ''}  {f'CLC {opts.clc_horizon}' if opts.clc_horizon > 0 else ''}"
        if opts.cost_discretization == "INTEGRATOR":
            label += " cost integration"
        labels.append(label)
    return labels


def run_closed_loop_experiment(n_runs: int, n_sim: int = 200):

    ref_ocp_opts = OcpOptions()

    reference, terminal_reference = setup_reference(dt_plant=ref_ocp_opts.dt0, n_sim=n_sim)

    reference_ocp_formulation = formulate_ocp(ref_ocp_opts, reference=reference)
    plant = AcadosIntegrator(reference_ocp_formulation)
    options_list = create_options_list()

    for ocp_options in options_list:
        id = dataclass_to_string(ocp_options)
        solver = setup_solver_from_opts(ocp_options, reference, terminal_reference)
        closed_loop_experiment(solver, plant, n_sim, X0, n_runs, id=id, print_stats_run_idx=[])
        solver = None


def evaluate_closed_loop_experiment(n_runs: int):

    # load results
    ref_ocp = formulate_ocp(OcpOptions())
    model = ref_ocp.model

    options_list = create_options_list()
    filenames = [
        get_results_filename(dataclass_to_string(ocp_opts), n_executions=n_runs)
        for ocp_opts in options_list
    ]
    labels = get_labels(options_list)
    results = get_results_from_filenames(filenames)

    print_closed_loop_costs_timings_table(results, labels)

    plot_trajectories(
        results,
        labels_list=labels,
        x_labels_list=model.x_labels,
        u_labels_list=model.u_labels,
        time_label=model.t_label,
        idxbu=ref_ocp.constraints.idxbu,
        lbu=ref_ocp.constraints.lbu,
        ubu=ref_ocp.constraints.ubu,
        ncol_legend=1,
        fig_filename="cstr_traj.pdf"
    )

    plot_acados_timings_submodules(results, labels, n_runs=n_runs)

    # pareto evaluation
    min_cost = np.min([r.cost_traj[-1] for r in results])
    rel_subopt = [100 * (r.cost_traj[-1] - min_cost) / min_cost for r in results]
    cpu_times = [1e3*np.mean(r.time_tot) for r in results]
    points = np.array([rel_subopt, cpu_times]).T

    varying_fields, _ = get_varying_fields(options_list)

    variants = {}
    for name in varying_fields:
        variants[name] = sorted(set([getattr(opts, name) for opts in options_list]))

    colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8']
    alphas = [1.0, 0.5, 0.3]
    markers = ['o', 'v', 's', 'd', '^', '>', '<', 'P', 'D', 'x', 'X']

    i_color = 1
    i_marker = 0
    i_alpha = 2
    n_varying = len(varying_fields)
    n_opts = len(options_list)

    markers_all = []
    colors_all = []
    alphas_all = []

    if i_color < n_varying:
        for opts in options_list:
            colors_all.append(colors[variants[varying_fields[i_color]].index(getattr(opts, varying_fields[i_color]))])
        color_legend = dict(zip(
                [f'{varying_fields[i_color]} {v}'
                for v in variants[varying_fields[i_color]]],
                colors))
    else:
        colors_all = n_opts * [colors[0]]
        color_legend = None

    if i_alpha < n_varying:
        for opts in options_list:
            alphas_all.append(alphas[variants[varying_fields[i_alpha]].index(getattr(opts, varying_fields[i_alpha]))])
        alpha_legend = dict(zip(
            [f'{varying_fields[i_alpha]} {v}'
            for v in variants[varying_fields[i_alpha]]],
            alphas))
    else:
        alphas_all = n_opts * [alphas[0]]
        alpha_legend = None

    if i_marker < n_varying:
        for opts in options_list:
            markers_all.append(markers[variants[varying_fields[i_marker]].index(getattr(opts, varying_fields[i_marker]))])
        marker_legend = dict(zip(
            [f'{varying_fields[i_marker]} {v}'
            for v in variants[varying_fields[i_marker]]],
            markers))
    else:
        markers_all = n_opts * [markers[0]]
        marker_legend = None


    plot_pareto(points,
                colors_all, alphas_all, markers_all,
                marker_legend=marker_legend,
                color_legend=color_legend,
                alpha_legend=alpha_legend,
                xlabel=r"Relative suboptimality [\%]",
                ylabel="CPU time [ms]", fig_filename="cstr_pareto.pdf")



if __name__ == "__main__":
    n_runs = 3
    run_closed_loop_experiment(n_runs)
    evaluate_closed_loop_experiment(n_runs)
