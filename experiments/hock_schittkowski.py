###############################################################################
# Solve Hock & Schittkowsky test problems with acados
###############################################################################
from dataclasses import dataclass
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel
import numpy as np
import casadi as ca
import os
import importlib
import sys
from contextlib import contextmanager
from mpc_solver_benchmark import AcadosSolver, single_ocp_experiment, get_results_filename, get_results_from_filenames, dataclass_to_string, hash_id, get_acados_branch_name, SimulationResultsOpenLoop

ACADOS_INF = 1e8
TOL = 1e-5
N_HORIZON = 1
N_RUNS = 1

@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

def get_hock_schittkowski_filenames():
    filenames = []
    for i in range(1, 120):
        filename = f'hs{i:03d}'
        if i == 87:
            # print(filename + '.....nonsmooth objective function...not in our scope')
            continue
        if i in [82, 94, 115]:
            # print(filename + '.....not existant...')
            continue
        filenames.append(filename)
    return filenames


@dataclass
class HockSchittkowskiSolverOptions:
    qp_solver: str = "FULL_CONDENSING_HPIPM"
    nlp_solver: str = "SQP"
    cost_discretization: str = "EULER"
    qp_solver_iter_max: int = 1000
    globalization: str = "FIXED_STEP"
    regularize_method: str = "MIRROR"
    nlp_solver_iter_max: int = 1000
    alpha_min: float = 1e-2
    line_search_use_sufficient_descent: int = 1
    eps_sufficient_descent: float = 1e-1

def get_hash(opts, name, branch_name):
    id = hash_id(dataclass_to_string(opts) + name + branch_name)
    return id

def solve_problem(filename, opts: HockSchittkowskiSolverOptions) -> None:
    # load problem
    prob = importlib.import_module(f'casadi_repo.test.python.hock_schittkowski.{filename}', package='mpc_solver_benchmark')
    hock_schittkowsky_func = getattr(prob, filename)
    with suppress_stdout():
        (x_opt, f_opt, x, f, g, lbg, ubg, lbx, ubx, x0) = hock_schittkowsky_func()

    branch_name = get_acados_branch_name()

    # create ocp
    ocp = create_acados_ocp(filename, x, f, g, lbg, ubg, lbx, ubx, opts)

    # create solver
    ocp_solver = AcadosSolver(ocp, with_x0=False)

    # initialize solver
    xinit = np.array(x0).squeeze()
    for i in range(N_HORIZON+1):
        ocp_solver.solver.set(i, "x", xinit)

    # solve ocp
    id = get_hash(opts, filename, branch_name)
    result = single_ocp_experiment(ocp_solver, x0=None, n_runs=N_RUNS, id=id, print_stats=False)

    # print summary
    print(f"cost function value = {ocp_solver.get_cost()} after {result.nlp_iter} SQP iterations")

    # compare to analytical solution
    sol_err = max(np.abs(result.x_traj[0] - x_opt))
    f_error = abs(f_opt - result.cost_value)

    print("Deviation from planned optimal solution: ", sol_err)
    if sol_err < TOL or f_error < TOL or result.status[0] == 0:
        print("Optimal solution found.")
    else:
        print("Not solved to required tolerance.")
    print("-----------------------------------")
    return


def create_acados_ocp(filename, x, f, g, lbg, ubg, lbx, ubx, opts: HockSchittkowskiSolverOptions) -> AcadosOcp:
    # replace inf with ACADOS_INF
    lbx = np.array(lbx).squeeze()
    lbx[lbx == -np.inf] = -ACADOS_INF
    ubx = np.array(ubx).squeeze()
    ubx[ubx == np.inf] = ACADOS_INF

    lbg = np.array(lbg).squeeze()
    lbg[lbg == -np.inf] = -ACADOS_INF
    ubg = np.array(ubg).squeeze()
    ubg[ubg == np.inf] = ACADOS_INF

    # create ocp object to formulate the OCP
    ocp = AcadosOcp()

    # dynamics: identity
    model = AcadosModel()
    model.disc_dyn_expr = x
    model.x = x
    model.u = ca.SX.sym('u', 0, 0) # [] / None doesnt work
    # model.p = []
    model.name = filename
    ocp.model = model

    # discretization
    Tf = 1
    ocp.dims.N = N_HORIZON
    ocp.solver_options.tf = Tf

    # cost
    ocp.cost.cost_type_e = 'EXTERNAL'
    ocp.model.cost_expr_ext_cost_e = f

    # constraints
    ocp.model.con_h_expr_0 = g
    ocp.constraints.lh_0 = np.array(lbg).squeeze()
    ocp.constraints.uh_0 = np.array(ubg).squeeze()

    ocp.constraints.lbx_0 = lbx
    ocp.constraints.ubx_0 = ubx
    ocp.constraints.idxbx_0 = np.arange(x.shape[0])

    # set options - fixed ones
    ocp.solver_options.integrator_type = 'DISCRETE'
    ocp.solver_options.hessian_approx = 'EXACT'
    ocp.solver_options.tol = TOL
    ocp.solver_options.qp_solver_iter_max = opts.qp_solver_iter_max
    ocp.solver_options.qp_tol = 0.5 * TOL
    # set options - variable ones
    ocp.solver_options.qp_solver = opts.qp_solver
    ocp.solver_options.nlp_solver_type = opts.nlp_solver
    ocp.solver_options.globalization = opts.globalization
    ocp.solver_options.regularize_method = opts.regularize_method
    ocp.solver_options.nlp_solver_max_iter = opts.nlp_solver_iter_max
    ocp.solver_options.alpha_min = opts.alpha_min
    ocp.solver_options.line_search_use_sufficient_descent = opts.line_search_use_sufficient_descent
    ocp.solver_options.eps_sufficient_descent = opts.eps_sufficient_descent
    return ocp


def run_benchmark_for_opts(opts: HockSchittkowskiSolverOptions):
    hs_filenames = get_hock_schittkowski_filenames()
    for name in hs_filenames:
        solve_problem(name, opts)


def evaluate_benchmark_run(opts: HockSchittkowskiSolverOptions, branch_name=None):
    hs_filenames = get_hock_schittkowski_filenames()

    if branch_name is None:
        branch_name = get_acados_branch_name()

    global_optimum_problem_ids = []
    unsolved_problem_ids = []
    kkt_point_problem_ids = []
    nlp_iter = []
    solver_time = []

    for name in hs_filenames:
        # load problem
        prob = importlib.import_module(f'casadi_repo.test.python.hock_schittkowski.{name}', package='mpc_solver_benchmark')
        hock_schittkowsky_func = getattr(prob, name)
        with suppress_stdout():
            (x_opt, f_opt, x, f, g, lbg, ubg, lbx, ubx, x0) = hock_schittkowsky_func()

        # load result
        id = get_hash(opts, name, branch_name)
        results_filename = get_results_filename(id=id, n_executions=N_RUNS)
        result: SimulationResultsOpenLoop = get_results_from_filenames([results_filename])[0]

        # compare to analytical solution
        sol_err = max(np.abs(result.x_traj[0] - x_opt))
        f_error = abs(f_opt - result.cost_value)
        if sol_err < TOL or (f_error < TOL and result.status[0] == 0):
            # print(f'{name} ..... solved to global optimum.')
            global_optimum_problem_ids.append(name)
        elif result.status[0] == 0:
            # print(f'{name} ..... solved to KKT point.')
            kkt_point_problem_ids.append(name)
        else:
            unsolved_problem_ids.append(name)
            # print(name + '.....not successful')

        # log stats
        nlp_iter.append(result.nlp_iter)
        solver_time.append(1e-3 * result.time_tot)

    print(f"Evaluating benchmark run for branch {branch_name} with options {opts}")
    print("\n-----------------------------------\n")
    print("Summary:")
    print(f"Out of {len(hs_filenames)}:")
    print(f"Problems solved to global optimality: {len(global_optimum_problem_ids)}")
    print(f"Problems solved to KKT point: {len(kkt_point_problem_ids)}")
    print(f"Unsolved problems: {len(unsolved_problem_ids)}")
    print("\n-----------------------------------\n")
    print(f"Number of NLP solver iterations: total: {sum(nlp_iter)}, average: {np.mean(nlp_iter):.2f}, mean: {np.median(nlp_iter)}")
    print(f"Solver time in [ms]: total: {sum(solver_time)}, average: {np.mean(solver_time):.3f}, mean: {np.median(solver_time):.3f}")
    print("\n-----------------------------------\n")
    print("list of unsolved problems: ", unsolved_problem_ids)


if __name__ == '__main__':
    opts = HockSchittkowskiSolverOptions()
    run_benchmark_for_opts(opts)
    evaluate_benchmark_run(opts, branch_name="master")
    evaluate_benchmark_run(opts, branch_name=None)
    # solve_problem('hs054')
    # solve_problem('hs003')
    # xinit = solve_problem_w_initial_guess('hs104')
    # solve_problem_w_initial_guess('hs104', xinit)
