from mpc_solver_benchmark import single_ocp_experiment, print_acados_timings_table_submodules, plot_acados_timings_submodules, get_results_filename, get_results_from_filenames, plot_trajectories, AcadosSolver, plot_trajectories_xy_space
from mpc_solver_benchmark.problems import formulate_unicycle_ocp, UnicycleOcpOptions, unicycle_get_circular_constraints
import numpy as np
import matplotlib.pyplot as plt

from acados_template import AcadosOcp, AcadosMultiphaseOcp


def get_id(options: UnicycleOcpOptions):
    return f"{options.model_type}"


def get_options_list():
    options_list = []
    options_list.append(UnicycleOcpOptions(
            cost_variant="HUBER",
            model_type="combined",
            N_horizon = 50,
            T_horizon = 5.,
            N_horizon_0= 20,
    ))
    for model_type in ["simplified_unicycle", "diff_drive"]:
        options_list.append(UnicycleOcpOptions(
            cost_variant="HUBER",
            model_type=model_type,
            N_horizon = 50,
            T_horizon = 5.
        ))

    return options_list

def experiment(n_runs: int):
    options = get_options_list()
    for opts in options:
        ocp = formulate_unicycle_ocp(opts)
        id = get_id(opts)

        solver = AcadosSolver(ocp)
        x0 = np.array([1.0, 1.0, 0.0, np.pi, 0.0])
        if opts.model_type == "simplified_unicycle":
            x0 = x0[:-1]
        single_ocp_experiment(solver, x0, n_runs, id=id)


def evaluate_experiment(n_runs: int):

    options_list = get_options_list()

    filenames = [get_results_filename(get_id(opts), n_executions=n_runs) for opts in options_list]

    labels = [f"{opts.model_type}" for opts in options_list]
    results = get_results_from_filenames(filenames)

    position_trajectories = []
    for res in results:
        position_trajectories.append( np.array([res.x_traj[i][:2] for i in range(len(res.x_traj))]) )

    circles = unicycle_get_circular_constraints()
    plot_trajectories_xy_space(position_trajectories=position_trajectories, labels=labels, circular_obstacles=circles)

    for result, label, opts in zip(results, labels, options_list):
        # if opts.model_type != "combined":
            ocp = formulate_unicycle_ocp(opts)
            if isinstance(ocp, AcadosOcp):
                model = ocp.model
                x_labels = model.x_labels
                u_labels = model.u_labels
                t_label = model.t_label
                idxpu = list(range(model.u.rows()))
                idxpx = list(range(model.x.rows()))
                idxbu = ocp.constraints.idxbu
                lbu = ocp.constraints.lbu
                ubu = ocp.constraints.ubu
            elif isinstance(ocp, AcadosMultiphaseOcp):
                model = ocp.model[0]
                x_labels = model.x_labels
                u_labels = model.u_labels
                t_label = model.t_label
                idxpu = []
                idxbu = []
                idxpx = [0, 1, 2, 3]
                lbu = []
                ubu = []

            plot_trajectories(
                [result],
                labels_list=[label],
                x_labels_list=x_labels,
                u_labels_list=u_labels,
                time_label=t_label,
                idxbu=idxbu,
                lbu=lbu,
                ubu=ubu,
                idxpu=idxpu,
                idxpx=idxpx
            )

    plot_acados_timings_submodules(
        results, labels, n_runs=n_runs, figure_filename=f"cost_comparison.png", per_iteration=True
    )
    print_acados_timings_table_submodules(results, labels)

    plt.show()



if __name__ == "__main__":
    n_runs = 1
    experiment(n_runs)
    evaluate_experiment(n_runs)
