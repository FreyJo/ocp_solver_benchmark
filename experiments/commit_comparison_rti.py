from mpc_solver_benchmark.problems import formulate_pendulum_ocp, PendulumOcpOptions
from mpc_solver_benchmark import AcadosSolver, get_results_filename, get_acados_branch_name, dataclass_to_string, single_ocp_experiment, print_acados_timings_table_submodules, plot_acados_timings_submodules, get_results_from_filenames

import numpy as np
import matplotlib.pyplot as plt
QP_SOLVER = "FULL_CONDENSING_DAQP"
REVISIONS = ["master", "dev_c"]


def experiment(n_runs: int):
    pendulum_ocp_options = PendulumOcpOptions(nlp_solver="SQP_RTI", qp_solver=QP_SOLVER)
    ocp = formulate_pendulum_ocp(pendulum_ocp_options)
    branch_name = get_acados_branch_name()
    id = f"{branch_name}_{dataclass_to_string(pendulum_ocp_options)}"
    print(f"Running experiment for branch {branch_name}")
    solver = AcadosSolver(ocp)
    x0 =  np.array([0.0, np.pi/10, 0.0, 0.0])
    single_ocp_experiment(solver, x0, n_runs, id=id)

def evaluate_experiment(n_runs: int):
    pendulum_ocp_options = PendulumOcpOptions(nlp_solver="SQP_RTI", qp_solver=QP_SOLVER, N_horizon=50)
    ids = [f"{branch_name}_{dataclass_to_string(pendulum_ocp_options)}" for branch_name in REVISIONS]
    filenames = [get_results_filename(id, n_executions=n_runs) for id in ids]
    results = get_results_from_filenames(filenames)
    labels = REVISIONS
    plot_acados_timings_submodules(
        results,
        labels,
        n_runs=n_runs,
        figure_filename=f"acados_{QP_SOLVER}_commit_comparison.png",
    )
    print(f"pendulum_ocp_options: {pendulum_ocp_options}")
    print_acados_timings_table_submodules(results, labels)
    plt.show()


if __name__ == "__main__":
    n_runs = 20000
    experiment(n_runs)
    evaluate_experiment(n_runs)
