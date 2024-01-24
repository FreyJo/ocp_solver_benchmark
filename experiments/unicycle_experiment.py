import numpy as np
import matplotlib.pyplot as plt

from mpc_solver_benchmark.problems import UnicycleOcpOptions, formulate_unicycle_ocp, unicycle_get_circular_constraints
X0 = np.array([1.0, 1.0, 0.0, np.pi, 0.0])

from mpc_solver_benchmark import (
    closed_loop_experiment,
    plot_acados_timings_submodules,
    get_results_filename,
    dataclass_to_string,
    AcadosSolver,
    AcadosIntegrator,
    print_closed_loop_costs_timings_table,
    get_results_from_filenames,
    plot_trajectories,
    plot_trajectories_xy_space,
)


def create_options_list():
    options_list = []
    N_horizon = 50
    # options_list.append(UnicycleOcpOptions(N_horizon=N_horizon, sim_method_num_steps=1, cost_variant="LINEAR_LS"))
    # options_list.append(UnicycleOcpOptions(N_horizon=N_horizon, sim_method_num_steps=1, cost_variant="HUBER", model_type="simplified_unicycle"))
    # globalization = "FIXED_STEP"
    globalization = "MERIT_BACKTRACKING"
    options_list.append(UnicycleOcpOptions(N_horizon=N_horizon, globalization=globalization, sim_method_num_steps=1, cost_variant="HUBER", model_type="diff_drive"))
    options_list.append(UnicycleOcpOptions(N_horizon=N_horizon, globalization=globalization, sim_method_num_steps=1, cost_variant="HUBER", model_type="combined", N_horizon_0=5))
    options_list.append(UnicycleOcpOptions(N_horizon=N_horizon, globalization=globalization, sim_method_num_steps=1, cost_variant="HUBER", model_type="combined", N_horizon_0=25))
    options_list.append(UnicycleOcpOptions(N_horizon=N_horizon, globalization=globalization, sim_method_num_steps=1, cost_variant="HUBER", model_type="combined", N_horizon_0=45))
    return options_list


def closed_loop_experiment_horizon(n_runs: int):
    reference_ocp_formulation = formulate_unicycle_ocp(UnicycleOcpOptions())
    plant = AcadosIntegrator(reference_ocp_formulation)
    options_list = create_options_list()
    for ocp_options in options_list:
        print(f"\nrunning simulation with {ocp_options=}\n")
        ocp = formulate_unicycle_ocp(ocp_options)
        solver = AcadosSolver(ocp)
        id = dataclass_to_string(ocp_options)

        # closed loop simulation
        n_sim = 200
        if ocp_options.model_type in ["diff_drive", "combined"]:
            x0 = X0
        elif ocp_options.model_type == "simplified_unicycle":
            x0 = X0[:-1]
        else:
            raise ValueError(f"Unknown model type {ocp_options.model_type}")

        closed_loop_experiment(
            solver, plant, n_sim, x0, n_runs, id=id, print_stats_run_idx=[]
        )
        solver = None


def get_labels(options_list: list[UnicycleOcpOptions]):
    labels = []
    for opts in options_list:
        label = f"$N={opts.N_horizon}$ {opts.model_type}"
        if opts.N_horizon_0 is not None:
            label += f" $N_0={opts.N_horizon_0}$"
        labels.append(label)
    return labels


def evaluate_closed_loop_experiment_horizon(n_runs: int):
    ref_ocp = formulate_unicycle_ocp(UnicycleOcpOptions())
    model = ref_ocp.model

    options = create_options_list()
    labels = get_labels(options)

    filenames = [
        get_results_filename(dataclass_to_string(ocp_opts), n_executions=n_runs)
        for ocp_opts in options
    ]
    results = get_results_from_filenames(filenames)

    position_trajectories = [res.x_traj[:, :2] for res in results]

    circles = unicycle_get_circular_constraints()
    plot_trajectories_xy_space(position_trajectories, labels,
                               circular_obstacles=circles)

    plot_trajectories(
        results,
        labels_list=labels,
        x_labels_list=model.x_labels,
        u_labels_list=model.u_labels,
        time_label=model.t_label,
        idxbu=ref_ocp.constraints.idxbu,
        lbu=ref_ocp.constraints.lbu,
        ubu=ref_ocp.constraints.ubu,
    )
    print_closed_loop_costs_timings_table(results,
                            labels,
                            )


    # plot_acados_timings_submodules(results, labels, n_runs=n_runs)
    plt.show()


if __name__ == "__main__":
    n_runs = 1
    closed_loop_experiment_horizon(n_runs)
    evaluate_closed_loop_experiment_horizon(n_runs)
