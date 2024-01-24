from mpc_solver_benchmark import single_ocp_experiment, print_acados_timings_table_submodules, plot_acados_timings_submodules, get_results_filename, get_results_from_filenames, plot_trajectories, AcadosSolver
from mpc_solver_benchmark.problems import formulate_pendulum_ocp, PendulumOcpOptions
import numpy as np
import matplotlib.pyplot as plt

QP_SOLVER = "FULL_CONDENSING_DAQP"
COST_VARIANTS = ["LINEAR_LS", "NONLINEAR_LS", "CONL", "EXTERNAL"]


def get_id(cost_variant: str):
    return f"{cost_variant}_{QP_SOLVER}"


def experiment(n_runs: int):
    for cost_variant in COST_VARIANTS:
        pendulum_ocp_options = PendulumOcpOptions(
            qp_solver=QP_SOLVER,
            nlp_solver="SQP",
            cost_variant=cost_variant,
            N_horizon = 20,
        )
        ocp = formulate_pendulum_ocp(pendulum_ocp_options)
        id = get_id(cost_variant)
        print(f"Running experiment for {cost_variant}")

        solver = AcadosSolver(ocp)
        x0 =  np.array([0.0, np.pi/10, 0.0, 0.0])
        single_ocp_experiment(solver, x0, n_runs, id=id)


def evaluate_experiment(n_runs: int):
    ids = [get_id(branch_name) for branch_name in COST_VARIANTS]
    filenames = [get_results_filename(id, n_executions=n_runs) for id in ids]
    labels = COST_VARIANTS
    results = get_results_from_filenames(filenames)
    plot_acados_timings_submodules(
        results, labels, n_runs=n_runs, figure_filename=f"cost_comparison.png"
    )
    print_acados_timings_table_submodules(results, labels)

    ref_ocp = formulate_pendulum_ocp(PendulumOcpOptions())
    model = ref_ocp.model

    plot_trajectories(
        results,
        labels_list=labels,
        x_labels_list=model.xlabels,
        u_labels_list=model.ulabels,
        time_label=model.time_label,
        idxbu=ref_ocp.constraints.idxbu,
        lbu=ref_ocp.constraints.lbu,
        ubu=ref_ocp.constraints.ubu,
    )

    plt.show()


if __name__ == "__main__":
    n_runs = 5
    experiment(n_runs)
    evaluate_experiment(n_runs)
