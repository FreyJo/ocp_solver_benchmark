import sys
import os

# getting the name of the directory
# where the this file is present.
current = os.path.dirname(os.path.realpath(__file__))

# Getting the parent directory name
# where the current directory is present.
parent = os.path.dirname(current)

# adding the parent directory to
# the sys.path.
sys.path.append(parent)
from mpc_solver_benchmark.problems.pendulum import formulate_time_optimal_swing_up, initialize_time_optimal_swing_up
from acados_template import AcadosOcpOptions
import numpy as np
from mpc_solver_benchmark import AcadosSolver, single_ocp_experiment, plot_acados_timings_submodules, get_results_filename, dataclass_to_string, get_results_from_filenames


def solve_single_ocp(opts):
    ocp = formulate_time_optimal_swing_up(opts)
    solver = AcadosSolver(ocp, with_x0=False)
    initialize_time_optimal_swing_up(solver.solver)
    single_ocp_experiment(solver, None, 1, id='1')

if __name__ == "__main__":
    # Move this to file
    opts = AcadosOcpOptions()
    opts.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
    opts.qp_solver_cond_N = 1
    opts.qp_solver_mu0 = 1e4
    opts.hessian_approx = 'EXACT'
    opts.integrator_type = 'ERK'
    opts.print_level = 1
    opts.N_horizon = 1
    opts.tf = 1
    opts.nlp_solver_max_iter = 500
    opts.regularize_method = 'MIRROR'
    opts.nlp_solver_type = 'SQP_WITH_FEASIBLE_QP'
    opts.globalization = 'FUNNEL_L1PEN_LINESEARCH'
    opts.globalization_funnel_use_merit_fun_only = False
    opts.qp_scaling_type = 'OBJECTIVE_GERSHGORIN'

    solve_single_ocp(opts)