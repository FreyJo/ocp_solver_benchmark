from mpc_solver_benchmark.problems import formulate_pendulum_ocp, PendulumOcpOptions
from mpc_solver_benchmark import hash_id, AcadosSolver, get_results_filename, dataclass_to_string, single_ocp_experiment, get_results_from_filenames
import matplotlib.pyplot as plt
import numpy as np

QP_SOLVER = "FULL_CONDENSING_DAQP"

OCP_OPTS = PendulumOcpOptions(nlp_solver="SQP", qp_solver=QP_SOLVER,
                sim_method_num_stages=4, sim_method_num_steps=10, integrator_type='IRK')


def get_id(with_openmp: bool, opts):
    return f"with_openmp_{with_openmp}_{hash_id(dataclass_to_string(opts))}"


def experiment(n_runs: int):

    ocp = formulate_pendulum_ocp(OCP_OPTS)
    solver = AcadosSolver(ocp)

    id = get_id(solver.solver.acados_lib_uses_omp, OCP_OPTS)

    print(f"Running experiment {id=}")
    x0 =  np.array([0.0, np.pi/10, 0.0, 0.0])
    single_ocp_experiment(solver, x0, n_runs, id=id)


def evaluate_experiment(n_runs: int):
    ids = [get_id(with_openmp, OCP_OPTS) for with_openmp in [True, False]]
    filenames = [get_results_filename(id, n_executions=n_runs) for id in ids]
    results = get_results_from_filenames(filenames)
    labels = ['with openmp', 'without openmp']

    for res, label in zip(results, labels):
        print(label, f'\tqp = {res.time_qp/1000:.2f}ms, \t remaining {(res.time_py-res.time_qp)/1000.:.2f}ms')

    plt.show()


if __name__ == "__main__":
    n_runs = 500
    experiment(n_runs)
    evaluate_experiment(n_runs)


