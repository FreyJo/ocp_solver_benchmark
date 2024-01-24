import numpy as np


PROBLEM_NAME = "cstr"
# PROBLEM_NAME = "pendulum"
PROBLEM_NAME = "unicycle"

if PROBLEM_NAME == "pendulum":
    from mpc_solver_benchmark.problems import formulate_pendulum_ocp as formulate_ocp
    from mpc_solver_benchmark.problems import PendulumOcpOptions as OcpOptions

    X0 = np.array([0.0, np.pi / 10, 0.0, 0.0])

elif PROBLEM_NAME == "cstr":
    from mpc_solver_benchmark.problems import formulate_cstr_ocp as formulate_ocp
    from mpc_solver_benchmark.problems import CstrOcpOptions as OcpOptions

    X0 = np.array([4.39000e-02, 2.43375e02, 3.29500e-01])

elif PROBLEM_NAME == "unicycle":
    from mpc_solver_benchmark.problems import formulate_unicycle_ocp as formulate_ocp
    from mpc_solver_benchmark.problems import UnicycleOcpOptions as OcpOptions

    X0 = np.array([1.0, 1.0, 0.0, 0.0, 0.0])

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
)

N_HORIZON_VALUES = [50]
N_HORIZON_VALUES_LABELS = [f"N={N_horizon}" for N_horizon in N_HORIZON_VALUES]


def create_options_list():
    options_list = []
    for N_horizon in N_HORIZON_VALUES:
        options_list.append(OcpOptions(N_horizon=N_horizon, sim_method_num_steps=1))
    return options_list


def closed_loop_experiment_horizon(n_runs: int):
    reference_ocp_formulation = formulate_ocp(OcpOptions())
    plant = AcadosIntegrator(reference_ocp_formulation)
    options_list = create_options_list()
    for ocp_options in options_list:
        ocp = formulate_ocp(ocp_options)
        solver = AcadosSolver(ocp)
        id = dataclass_to_string(ocp_options)

        # closed loop simulation
        n_sim = 100
        closed_loop_experiment(
            solver, plant, n_sim, X0, n_runs, id=id, print_stats_run_idx=[]
        )
        solver = None


def evaluate_closed_loop_experiment_horizon(n_runs: int):
    ref_ocp = formulate_ocp(OcpOptions())
    model = ref_ocp.model

    filenames = [
        get_results_filename(dataclass_to_string(ocp_opts), n_executions=n_runs)
        for ocp_opts in create_options_list()
    ]
    results = get_results_from_filenames(filenames)
    labels = N_HORIZON_VALUES_LABELS

    print_closed_loop_costs_timings_table(results, labels)
    plot_trajectories(
        results,
        labels_list=N_HORIZON_VALUES_LABELS,
        x_labels_list=model.x_labels,
        u_labels_list=model.u_labels,
        time_label=model.t_label,
        idxbu=ref_ocp.constraints.idxbu,
        lbu=ref_ocp.constraints.lbu,
        ubu=ref_ocp.constraints.ubu,
    )

    plot_acados_timings_submodules(results, labels, n_runs=n_runs)


if __name__ == "__main__":
    n_runs = 1
    closed_loop_experiment_horizon(n_runs)
    evaluate_closed_loop_experiment_horizon(n_runs)
