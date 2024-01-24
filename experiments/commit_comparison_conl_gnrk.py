import numpy as np

from mpc_solver_benchmark import AcadosSolver, single_ocp_experiment, print_acados_timings_table_submodules, plot_acados_timings_submodules, get_results_filename, get_acados_branch_name, hash_id, dataclass_to_string, get_results_from_filenames
from mpc_solver_benchmark.problems import formulate_pendulum_ocp, PendulumOcpOptions

REVISIONS = ["master", "diag_flag"]


TEST_PROBLEM_OPTIONS = PendulumOcpOptions(
    qp_solver="FULL_CONDENSING_DAQP",
    cost_variant="CONL_LARGE",
    # cost_variant="CONL",
    nlp_solver="SQP_RTI",
    cost_discretization="INTEGRATOR",
    integrator_type="IRK",
    N_horizon=50
)

def experiment(n_runs: int):

    ocp = formulate_pendulum_ocp(TEST_PROBLEM_OPTIONS)

    branch_name = get_acados_branch_name()
    id = branch_name + str(hash_id(dataclass_to_string(TEST_PROBLEM_OPTIONS)))
    print(f"Running experiment for branch {branch_name}")

    solver = AcadosSolver(ocp)
    x0 =  np.array([0.0, np.pi/10, 0.0, 0.0])
    single_ocp_experiment(solver, x0, n_runs, id=id)


def evaluate_experiment(n_runs: int):
    ids = [branch_name + str(hash_id(dataclass_to_string(TEST_PROBLEM_OPTIONS))) for branch_name in REVISIONS]
    filenames = [get_results_filename(id, n_executions=n_runs) for id in ids]
    results = get_results_from_filenames(filenames)
    labels = REVISIONS
    plot_acados_timings_submodules(
        results,
        labels,
        n_runs=n_runs,
        figure_filename=f"cost_commit_comparison.png",
    )
    print_acados_timings_table_submodules(results, labels)


if __name__ == "__main__":
    n_runs = 5000
    experiment(n_runs)
    evaluate_experiment(n_runs)
