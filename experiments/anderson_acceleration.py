from mpc_solver_benchmark import (single_ocp_experiment,
                                  print_acados_timings_table_submodules,
                                  plot_acados_timings_submodules,
                                  get_results_filename,
                                  get_results_from_filenames,
                                  plot_trajectories,
                                  AcadosSolver,
                                  plot_contraction_rate)
from mpc_solver_benchmark.problems import formulate_crane_ocp, CraneOcpOptions
import numpy as np
import matplotlib.pyplot as plt

QP_SOLVER = "PARTIAL_CONDENSING_HPIPM"


def get_id(with_ac: str):
    return f"AC_{with_ac}_{QP_SOLVER}"

def init_solver(solver: AcadosSolver, params: CraneOcpOptions):
    # initialization
    u_init_traj = np.tile(np.expand_dims(np.array([0.1, 0.5]), axis=1), reps=params.N_horizon)
    x_init_traj = np.tile(np.expand_dims(params.x0, axis=1), reps=params.N_horizon+1)*(1.-np.linspace(0, 1, params.N_horizon+1)) + \
                  np.tile(np.expand_dims(params.xf, axis=1), reps=params.N_horizon+1)*np.linspace(0, 1, params.N_horizon+1)

    solver.set_x_init_traj(x_init_traj.T)
    solver.set_u_init_traj(u_init_traj.T)

    return solver


def experiment(n_runs: int):

    for with_ac in [True, False]:
        crane_ocp_options = CraneOcpOptions(
            with_anderson_acceleration=with_ac,
            qp_solver=QP_SOLVER
        )
        ocp = formulate_crane_ocp(crane_ocp_options)
        id = get_id(with_ac)
        print(f"Running experiment with Anderson Acceleration {with_ac}")

        solver = AcadosSolver(ocp)
        init_solver(solver, crane_ocp_options)
        single_ocp_experiment(solver, crane_ocp_options.x0, n_runs, id=id, log_step_norm=True)
        solver = None


def evaluate_experiment(n_runs: int):

    crane_ocp_options = CraneOcpOptions(qp_solver=QP_SOLVER)
    ocp = formulate_crane_ocp(crane_ocp_options)
    model = ocp.model

    ids = [get_id(with_ac) for with_ac in [True, False]]
    filenames = [get_results_filename(id, n_executions=n_runs) for id in ids]
    labels = ['with AC', 'without AC']
    results = get_results_from_filenames(filenames)
    plot_acados_timings_submodules(
        results, labels, n_runs=n_runs, figure_filename=f"anderson_acceleration.png"
    )
    print_acados_timings_table_submodules(results, labels)

    plot_trajectories(
        results,
        labels_list=labels,
        x_labels_list=model.x_labels,
        u_labels_list=model.u_labels,
        time_label=model.t_label,
        idxbu=ocp.constraints.idxbu,
        lbu=ocp.constraints.lbu,
        ubu=ocp.constraints.ubu,
    )

    plot_contraction_rate(results, labels_list=labels)

    plt.show()


if __name__ == "__main__":
    n_runs = 10
    experiment(n_runs)
    evaluate_experiment(n_runs)
