from typing import List, Optional
import numpy as np
from acados_template import AcadosSimSolver
from .utils import get_results_filename
from .simulation_results import SimulationResultsOpenLoop, SimulationResultsClosedLoop
from .solver_base import BaseSolver
import time

def single_ocp_experiment(solver: BaseSolver, x0: Optional[np.ndarray], n_runs: int, id: str, log_step_norm: bool = False, print_stats: bool = True):

    results_filename = get_results_filename(id, n_executions=n_runs)

    time_tot = np.zeros((n_runs,))
    time_lin = np.zeros((n_runs,))
    time_sim = np.zeros((n_runs,))
    time_qp = np.zeros((n_runs,))
    time_qp_solver_call = np.zeros((n_runs,))
    time_reg = np.zeros((n_runs,))
    time_py = np.zeros((n_runs,))

    solver.store_iterate("solver_init.json", overwrite=True)
    for j in range(n_runs):
        solver.reset()
        solver.load_iterate("solver_init.json")
        t0 = time.time()
        if x0 is not None:
            _ = solver.solve_for_x0(x0=x0, t=0.)
        else:
            _ = solver.solve()

        time_py[j] = time.time() - t0
        time_tot[j] = solver.get_stats("time_tot")
        time_lin[j] = solver.get_stats("time_lin")
        time_sim[j] = solver.get_stats("time_sim")
        time_qp[j] = solver.get_stats("time_qp")
        time_qp_solver_call[j] = solver.get_stats("time_qp_solver_call")
        time_reg[j] = solver.get_stats("time_reg")

        if j == n_runs - 1 and print_stats:
            solver.print_statistics()

    y_ref_traj, y_ref_terminal = solver.get_y_reference_traj()

    results = SimulationResultsOpenLoop(
        time_stamp = str(time.time()),
        time_tot = 1e6 * np.min(time_tot),
        time_sim = 1e6 * np.min(time_sim),
        time_lin = 1e6 * np.min(time_lin),
        time_qp_solver_call = 1e6 * np.min(time_qp_solver_call),
        time_qp = 1e6 * np.min(time_qp),
        time_reg = 1e6 * np.min(time_reg),
        time_py = 1e6 * np.min(time_py),
        x_traj=solver.get_x_traj(),
        u_traj=solver.get_u_traj(),
        y_ref_traj=y_ref_traj,
        y_ref_terminal=y_ref_terminal,
        t_traj=solver.get_t_traj(),
        nlp_iter = solver.get_stats("nlp_iter"),
        primal_step_norm_traj= None if not log_step_norm else solver.get_stats("primal_step_norm"),
        cost_value = solver.get_cost(),
        status = np.array([solver.get_status()]),
    )

    results.save_to_file(filename=results_filename)
    return results


def closed_loop_experiment(solver: BaseSolver,
                           plant: AcadosSimSolver,
                           n_sim: int,
                           x0: np.ndarray,
                           n_runs: int,
                           id: str,
                           print_stats_run_idx: Optional[List[int]] = None,
                           u_disturbance: Optional[np.ndarray] = None,
                           print_stats_on_failure = True,
                           print_stats_on_solver_max_iter = False
                           ):

    results_filename = get_results_filename(id, n_executions=n_runs)

    time_tot = np.zeros((n_runs, n_sim))
    time_lin = np.zeros((n_runs, n_sim))
    time_sim = np.zeros((n_runs, n_sim))
    time_qp = np.zeros((n_runs, n_sim))
    time_qp_solver_call = np.zeros((n_runs, n_sim))
    time_reg = np.zeros((n_runs, n_sim))
    time_preparation = np.zeros((n_runs, n_sim))
    time_feedback = np.zeros((n_runs, n_sim))

    cost_traj = np.zeros((n_sim,))
    nlp_iter = np.zeros((n_sim,))
    qp_iter = np.zeros((n_sim,))
    x_traj = np.zeros((n_sim+1, x0.shape[0]))
    x_traj[0, :] = x0
    nu = solver.get_nu()
    u_traj = np.zeros((n_sim, nu))

    if solver.has_reference:
        y_ref_traj = np.zeros((n_sim, solver.reference.ny))
    else:
        y_ref_traj = None

    if print_stats_run_idx is None:
        print_stats_run_idx = []

    solver.store_iterate("solver_init.json", overwrite=True)
    for j in range(n_runs):
        solver.reset()
        solver.load_iterate("solver_init.json")
        x_current = x0
        c_current = 0.0
        t_current = 0.0
        for i in range(n_sim):
            if not solver.is_real_time_solver:
                u = solver.solve_for_x0(x_current, t_current)
            else:
                solver.real_time_preparation(t_current)
                u = solver.real_time_feedback(x_current, t_current)

            status = solver.get_status()
            if j in print_stats_run_idx or \
                (print_stats_on_solver_max_iter and status == 2) or \
                (print_stats_on_failure and status not in [0, 2]):
                solver.print_statistics()

            # if status != 0 and j == 0:
                # print(f"got status {status} in simulation step {i}.")
                # breakpoint()
                # pass

            if u_disturbance is not None:
                u += u_disturbance[i, :]

            x_augmented_current = plant.simulate(
                np.concatenate((x_current, np.array([c_current]))),
                u,
                t=np.array([t_current]),
            )

            x_current = x_augmented_current[:-1]
            c_current = x_augmented_current[-1]

            time_tot[j, i] = solver.get_stats("time_tot")
            time_lin[j, i] = solver.get_stats("time_lin")
            time_sim[j, i] = solver.get_stats("time_sim")
            time_qp[j, i] = solver.get_stats("time_qp")
            time_qp_solver_call[j, i] = solver.get_stats("time_qp_solver_call")
            time_reg[j, i] = solver.get_stats("time_reg")
            time_preparation[j, i] = solver.get_stats("time_preparation")
            time_feedback[j, i] = solver.get_stats("time_feedback")

            if j == 0:
                nlp_iter[i] = solver.get_stats("nlp_iter")
                qp_iter[i] = sum(solver.get_stats("qp_iter"))
                x_traj[i+1, :] = x_current
                u_traj[i, :] = u
                cost_traj[i] = c_current

                if solver.has_reference:
                    y_ref_traj[i, :] = solver.reference.get_reference(t_current).flatten()

            t_current += plant.T


    results = SimulationResultsClosedLoop(
        time_stamp = str(time.time()),
        time_tot = 1e6 * np.min(time_tot, axis=0),
        time_sim = 1e6 * np.min(time_sim, axis=0),
        time_lin = 1e6 * np.min(time_lin, axis=0),
        time_qp_solver_call = 1e6 * np.min(time_qp_solver_call, axis=0),
        time_qp = 1e6 * np.min(time_qp, axis=0),
        time_reg = 1e6 * np.min(time_reg, axis=0),
        time_preparation = 1e6 * np.min(time_preparation, axis=0),
        time_feedback = 1e6 * np.min(time_feedback, axis=0),
        nlp_iter = nlp_iter,
        qp_iter=qp_iter,
        cost_traj = cost_traj,
        x_traj=x_traj,
        u_traj=u_traj,
        y_ref_traj=y_ref_traj,
        t_traj=plant.T*np.arange(n_sim+1),
    )
    results.save_to_file(results_filename)
