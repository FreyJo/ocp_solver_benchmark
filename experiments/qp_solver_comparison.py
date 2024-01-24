from mpc_solver_benchmark.problems import formulate_pendulum_ocp, PendulumOcpOptions
from mpc_solver_benchmark import AcadosSolver, single_ocp_experiment, plot_acados_timings_submodules, get_results_filename, dataclass_to_string, get_results_from_filenames

import numpy as np

QP_SOLVERS = [
    "PARTIAL_CONDENSING_HPIPM",
    "FULL_CONDENSING_QPOASES",
    "FULL_CONDENSING_HPIPM",
    "FULL_CONDENSING_DAQP",
]

QP_SOLVER_LABELS = [name.replace("_", " ") for name in QP_SOLVERS]


def qp_solver_experiment(n_runs: int):
    # create ocp object to formulate the OCP
    for qp_solver in QP_SOLVERS:
        pendulum_ocp_options = PendulumOcpOptions(
            nlp_solver="SQP",
            qp_solver=qp_solver,
        )
        ocp = formulate_pendulum_ocp(pendulum_ocp_options)
        id = dataclass_to_string(pendulum_ocp_options)
        solver = AcadosSolver(ocp)
        x0 =  np.array([0.0, np.pi/10, 0.0, 0.0])
        single_ocp_experiment(solver, x0, n_runs, id=id)


def evaluate_qp_solver_experiment(n_runs: int):
    ids = [dataclass_to_string(PendulumOcpOptions(qp_solver=qp_solver)) for qp_solver in QP_SOLVERS]

    filenames = [
        get_results_filename(id, n_executions=n_runs) for id in ids
    ]
    results = get_results_from_filenames(filenames)
    labels = QP_SOLVER_LABELS
    plot_acados_timings_submodules(results, labels, n_runs=n_runs)


if __name__ == "__main__":
    n_runs = 500
    qp_solver_experiment(n_runs)
    evaluate_qp_solver_experiment(n_runs)
