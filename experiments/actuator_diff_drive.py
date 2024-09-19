from mpc_solver_benchmark import get_results_filename, get_results_from_filenames, plot_trajectories, AcadosSolver, plot_trajectories_xy_space, AcadosIntegrator, SimulationResultsClosedLoop, closed_loop_experiment, print_closed_loop_costs_timings_table, dataclass_to_string, plot_pareto, get_varying_fields, hash_id
from mpc_solver_benchmark.problems import formulate_unicycle_ocp, UnicycleOcpOptions, unicycle_get_circular_constraints
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

X0 = np.array([1.0, 1.0, 0.0, np.pi, 0.0, 0.0, 0.0])

def get_options_list():
    options_list = []
    N_horizon = 60
    options_list.append(UnicycleOcpOptions(
            cost_variant="LINEAR_LS",
            model_type="actuators",
            cost_discretization="EULER",
            globalization="FIXED_STEP",
            N_horizon = 60,
            T_horizon = 5.,
    ))
    for nlp_solver_type in ["SQP", "SQP_RTI"]:
        for N_horizon in [10, 20, 30, 40, 50]:
            options_list.append(UnicycleOcpOptions(
                    cost_variant="LINEAR_LS",
                    model_type="actuators",
                    cost_discretization="EULER",
                    globalization="FIXED_STEP",
                    N_horizon = N_horizon,
                    T_horizon = N_horizon/10,
                    nlp_solver_type=nlp_solver_type,
            ))

        N_horizon = 50
        for N_horizon_0 in [10, 20, 30, 40]:
            options_list.append(UnicycleOcpOptions(
                cost_variant="LINEAR_LS",
                model_type="actuators_to_diff_drive",
                cost_discretization="EULER",
                globalization="FIXED_STEP",
                N_horizon = N_horizon,
                N_horizon_0 = N_horizon_0,
                T_horizon = 5.,
                nlp_solver_type=nlp_solver_type,
            ))

        N_horizon = 40
        for N_horizon_0 in [10, 20, 30]:
            options_list.append(UnicycleOcpOptions(
                cost_variant="LINEAR_LS",
                model_type="actuators_to_diff_drive",
                cost_discretization="EULER",
                globalization="FIXED_STEP",
                N_horizon = N_horizon,
                N_horizon_0 = N_horizon_0,
                T_horizon = 5.,
                nlp_solver_type=nlp_solver_type,
            ))

    return options_list


def closed_loop_experiment_diff_drive(n_runs: int):
    reference_ocp_formulation = formulate_unicycle_ocp(UnicycleOcpOptions(model_type="actuators", cost_variant="LINEAR_LS", sim_method_num_steps=10))
    plant = AcadosIntegrator(reference_ocp_formulation)
    options_list = get_options_list()
    for ocp_options in options_list:
        print(f"\nrunning simulation with {ocp_options=}\n")
        ocp = formulate_unicycle_ocp(ocp_options)
        solver = AcadosSolver(ocp)
        id = hash_id(dataclass_to_string(ocp_options))

        # closed loop simulation
        n_sim = 200
        x0 = X0

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
        label += f" {opts.nlp_solver_type}"
        labels.append(label)
    return labels


def evaluate_closed_loop_experiment_diff_drive(n_runs: int):
    ref_ocp = formulate_unicycle_ocp(UnicycleOcpOptions(model_type="actuators", cost_variant="LINEAR_LS"))
    model = ref_ocp.model

    options = get_options_list()
    labels = get_labels(options)

    filenames = [
        get_results_filename(hash_id(dataclass_to_string(ocp_opts)), n_executions=n_runs)
        for ocp_opts in options
    ]
    results: list[SimulationResultsClosedLoop] = get_results_from_filenames(filenames)

    position_trajectories = [res.x_traj[:, :2] for res in results]

    circles = unicycle_get_circular_constraints()

    plot_traj = False
    if plot_traj:
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

    ## pareto plot
    min_cost = np.min([r.cost_traj[-1] for r in results])
    rel_subopt = [100 * (r.cost_traj[-1] - min_cost) / min_cost for r in results]
    # rel_subopt = [100 * (r.cost_traj[-1]) for r in results]

    cpu_times = [1e-3*np.mean(r.time_tot/r.nlp_iter) for r in results]
    ylabel = "computation time per iter. [ms]"
    cpu_times = [1e-3*np.mean(r.time_tot) for r in results]
    ylabel = "mean computation time [ms]"

    points = np.array([rel_subopt, cpu_times]).T

    # remove reference point
    points = points[1:, :]
    options = options[1:]

    varying_fields, _ = get_varying_fields(options)
    varying_fields.remove("model_type")
    varying_fields.remove("T_horizon")

    variants = {}
    for name in varying_fields:
        variant_set = set([getattr(opts, name) for opts in options])
        if None in variant_set:
            variant_set.remove(None)
            variants[name] = sorted(variant_set)
            variants[name].append(None)
        else:
            variants[name] = sorted(variant_set)

    print(f"{varying_fields=}")
    alphas = np.arange(1., 0.1, -0.1)

    i_alpha = 5
    n_opts = len(options)

    alphas_all = []
    colors_all = []
    markers_all = []
    markersizes_all = []

    variants["N_horizon_0"].remove(None)
    # variants['N_horizon'] = sorted(set(variants["N_horizon"] + variants["N_horizon_0"]), reverse=True)
    variants['N_horizon'] = sorted(variants["N_horizon"], reverse=True)
    print(f"{variants['N_horizon']=}")
    print(f"{variants['N_horizon_0']=}")
    cmaps = [matplotlib.colormaps["Oranges"], matplotlib.colormaps["Blues"], matplotlib.colormaps["Greens"], matplotlib.colormaps["Purples"]]

    def get_float_from_0tomax_interval(max, entry):
        out = entry / max
        return out

    color_legend = dict()
    for opts in options:
        if opts.N_horizon_0 is None:
            # colors_all.append('C1')
            color = cmaps[0](get_float_from_0tomax_interval(max(variants["N_horizon"]), opts.N_horizon))
            colors_all.append(color)
            legend_key =r'OCP $N = ' + f"{opts.N_horizon}$"
        else: # MOCP
            i_cmap = variants['N_horizon'].index(opts.N_horizon) + 1
            color = cmaps[i_cmap](get_float_from_0tomax_interval(opts.N_horizon, opts.N_horizon_0))
            colors_all.append(color)
            legend_key = r'MOCP $N= ' + f"{opts.N_horizon}$, " + r'$N_{\mathrm{act}} = ' + f"{opts.N_horizon_0}$"

        color_legend[legend_key] = color
        markers_all.append('o' if opts.nlp_solver_type == "SQP" else 'v')

    markersize_legend = None
    markersizes_all = None
    marker_legend = {'SQP': 'o', 'RTI': 'v'}

    varying_fields.remove("N_horizon_0")
    n_varying = len(varying_fields)

    if i_alpha < n_varying:
        for opts in options:
            alphas_all.append(alphas[variants[varying_fields[i_alpha]].index(getattr(opts, varying_fields[i_alpha]))])
        alpha_legend = dict(zip(
            [f'{varying_fields[i_alpha]} {v}'
            for v in variants[varying_fields[i_alpha]]],
            alphas))
    else:
        alphas_all = n_opts * [alphas[0]]
        alpha_legend = None

    if 0:
        plot_pareto(points,
                    colors_all, alphas_all, markers_all,
                    marker_legend=marker_legend,
                    color_legend=color_legend,
                    alpha_legend=alpha_legend,
                    markersizes=markersizes_all,
                    markersize_legend=markersize_legend,
                    xlabel=r"Relative suboptimality [\%]",
                    xscale="log",
                    ncol_legend=2,
                    figsize=(9, 5),
                    ylabel=ylabel, fig_filename="diff_drive_pareto.pdf")
    else:
        for nlp_solver in ['SQP', 'SQP_RTI']:
            points_ = []
            c_ = []
            a_ = []
            m_ = []
            for i, opts in enumerate(options):
                if opts.nlp_solver_type == nlp_solver:
                    points_.append(points[i])
                    c_.append(colors_all[i])
                    a_.append(alphas_all[i])
                    m_.append(markers_all[i])

            if nlp_solver == 'SQP_RTI':
                with_legend = True
                xlim = [2, 100]
            else:
                with_legend = False
                xlim = [0.1, 1e2]

            plot_pareto(points_,
                    c_, a_, m_,
                    marker_legend=marker_legend if with_legend else None,
                    color_legend=color_legend if with_legend else None,
                    alpha_legend=alpha_legend if with_legend else None,
                    markersizes=markersizes_all if with_legend else None,
                    markersize_legend=markersize_legend,
                    xlabel=r"Relative suboptimality [\%]",
                    xscale="log",
                    ncol_legend=1,
                    title='RTI' if nlp_solver=="SQP_RTI" else nlp_solver,
                    figsize=(6.6, 5),
                    bbox_to_anchor=(1., 1.0),
                    ylabel=ylabel, fig_filename=f"diff_drive_pareto_{nlp_solver}.pdf",
                    xlim=xlim)
    plt.show()



if __name__ == "__main__":
    n_runs = 5
    # closed_loop_experiment_diff_drive(n_runs)
    evaluate_closed_loop_experiment_diff_drive(n_runs)
