import numpy as np
import matplotlib.pyplot as plt

PROBLEM_NAME = "pendulum"
# PROBLEM_NAME = "pendulum_rate"

if PROBLEM_NAME == "pendulum":
    from mpc_solver_benchmark.problems import formulate_pendulum_ocp as formulate_ocp
    from mpc_solver_benchmark.problems import PendulumOcpOptions as OcpOptions
    X0 = np.array([0.0, np.pi / 8, 0.0, 0.0])
elif PROBLEM_NAME == "pendulum_rate":
    from mpc_solver_benchmark.problems import formulate_pendulum_rate_ocp as formulate_ocp
    from mpc_solver_benchmark.problems import PendulumRateOcpOptions as OcpOptions
    X0 = np.array([0.0, np.pi / 8, 0.0, 0.0, 0.0])



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

N_HORIZON_VALUES = [40]
N_HORIZON_VALUES_LABELS = [f"N={N_horizon}" for N_horizon in N_HORIZON_VALUES]


def create_options_list():
    options_list = []
    for N_horizon in N_HORIZON_VALUES:
        options_list.append(OcpOptions(N_horizon=N_horizon, sim_method_num_steps=1, nlp_solver="SQP_RTI", integrator_type="IRK"))
    return options_list



def initialize_solver(solver: AcadosSolver, x0):
    x_init = np.linspace(x0, 0*x0, solver.ocp.solver_options.N_horizon+1)
    for i in range(solver.ocp.solver_options.N_horizon+1):
        solver.solver.set(i, "x", x_init[i, :].flatten())


def closed_loop_experiment_horizon(n_runs: int):
    reference_ocp_formulation = formulate_ocp(OcpOptions())
    plant = AcadosIntegrator(reference_ocp_formulation)
    options_list = create_options_list()
    for ocp_options in options_list:
        ocp = formulate_ocp(ocp_options)
        ocp.solver_options.levenberg_marquardt = 1e-6
        solver = AcadosSolver(ocp)
        id = dataclass_to_string(ocp_options)

        initialize_solver(solver, X0)

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
    plt.show()


if __name__ == "__main__":
    n_runs = 20
    closed_loop_experiment_horizon(n_runs)
    evaluate_closed_loop_experiment_horizon(n_runs)
