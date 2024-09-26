import os
import numpy as np

import matplotlib.pyplot as plt

from mpc_solver_benchmark.problems import formulate_zanelli_tightening_pendulum_ocp as formulate_ocp
from mpc_solver_benchmark.problems import ZanelliTighteningPendulumOcpOptions as OcpOptions

X0 = np.array([0.0, np.pi, 0.0, 0.0])

from mpc_solver_benchmark import (
    closed_loop_experiment,
    plot_acados_timings_submodules,
    plot_acados_timings_real_time_split,
    get_results_filename,
    dataclass_to_string,
    AcadosSolver,
    AcadosIntegrator,
    print_closed_loop_costs_timings_table,
    get_results_from_filenames,
    plot_trajectories,
    print_acados_timings_table_submodules,
    print_acados_timings_table_real_time_split,
)


def create_options_list():
    options_list = []
    qp_solver = "PARTIAL_CONDENSING_HPIPM"
    # qp_solver = "FULL_CONDENSING_DAQP"
    for (N_horizon, N_exact) in [(100, 5), (100, 10), (100, 20), (100, 50), (100, 100), (50, 50),]:
        options_list.append(OcpOptions(N_horizon=N_horizon, N_exact=N_exact, T_horizon=N_horizon*0.01, qp_solver=qp_solver))
    return options_list


def create_options_list_traj_plot():
    options_list = []
    qp_solver = "PARTIAL_CONDENSING_HPIPM"
    for (N_horizon, N_exact) in [(100, 5), (100, 20), (100, 100), (50, 50),]:
        options_list.append(OcpOptions(N_horizon=N_horizon, N_exact=N_exact, T_horizon=N_horizon*0.01, qp_solver=qp_solver))
    return options_list

def closed_loop_experiment_horizon(n_runs: int):
    reference_ocp_formulation = formulate_ocp(OcpOptions())
    plant = AcadosIntegrator(reference_ocp_formulation)
    options_list = create_options_list()
    for ocp_options in options_list:
        ocp = formulate_ocp(ocp_options)
        solver = AcadosSolver(ocp)
        # for i in range(ocp_options.N_horizon):
        #     solver.solver.set(i, "u", np.array([1.0]))

        id = dataclass_to_string(ocp_options)

        # closed loop simulation
        n_sim = 500
        closed_loop_experiment(
            solver, plant, n_sim, X0, n_runs, id=id, print_stats_run_idx=[]
        )
        solver = None

def get_label_from_opts(ocp_opts):
    # return f"N={ocp_opts.N_horizon}, N_exact={ocp_opts.N_exact}"
    return f"$N={ocp_opts.N_horizon}, " + "N_{\mathrm{exact}}= " + f"{ocp_opts.N_exact}$"

def evaluate_closed_loop_experiment_ptight_timings(n_runs: int):
    ref_ocp = formulate_ocp(OcpOptions())
    model = ref_ocp.model
    table_style = 'latex'

    opts_list = create_options_list()
    filenames = [
        get_results_filename(dataclass_to_string(ocp_opts), n_executions=n_runs)
        for ocp_opts in opts_list
    ]
    results = get_results_from_filenames(filenames)
    labels = [get_label_from_opts(ocp_opts) for ocp_opts in opts_list]

    print_closed_loop_costs_timings_table(results, labels, style=table_style)

    metric = 'max'
    plot_acados_timings_submodules(results, labels, n_runs=n_runs, metric=metric)
    plot_acados_timings_real_time_split(results, labels, n_runs=n_runs, metric=metric,
            fig_filename=os.path.join("figures", "zanelli_ptight_rti_timings.pdf"),
            figsize=(8, 4), title=" ",)
    print_acados_timings_table_real_time_split(results, labels, metric=metric, style=table_style, include_relative_suboptimality=True)
    print_acados_timings_table_submodules(results, labels, metric=metric, style=table_style)
    plt.show()


def evaluate_closed_loop_experiment_ptight_trajectories(n_runs: int):

    ref_ocp = formulate_ocp(OcpOptions())
    model = ref_ocp.model
    table_style = 'latex'

    opts_list = create_options_list_traj_plot()
    filenames = [
        get_results_filename(dataclass_to_string(ocp_opts), n_executions=n_runs)
        for ocp_opts in opts_list
    ]
    results = get_results_from_filenames(filenames)
    labels = [get_label_from_opts(ocp_opts) for ocp_opts in opts_list]

    plot_trajectories(
        results,
        labels_list=labels,
        x_labels_list=model.x_labels,
        u_labels_list=model.u_labels,
        time_label=model.t_label,
        idxbu=ref_ocp.constraints.idxbu,
        lbu=ref_ocp.constraints.lbu,
        ubu=ref_ocp.constraints.ubu,
        idxpx=[0, 1],
        figsize=(12, 4),
        fig_filename=os.path.join("figures", "zanelli_ptight_trajectories.pdf"),
        x_max=[4.0, None],
        bbox_to_anchor=(0.5, -1.0),
    )

    plt.show()


if __name__ == "__main__":
    n_runs = 40
    # closed_loop_experiment_horizon(n_runs)
    evaluate_closed_loop_experiment_ptight_timings(n_runs)
    evaluate_closed_loop_experiment_ptight_trajectories(n_runs)
