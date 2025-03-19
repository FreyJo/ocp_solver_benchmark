#
# Copyright (c) The acados authors.
#
# This file is part of acados.
#
# The 2-Clause BSD License
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.;
#

from typing import Optional

import numpy as np
import casadi as ca
import scipy

from dataclasses import dataclass

from acados_template import AcadosOcp, AcadosModel, AcadosMultiphaseOcp

from .pendulum import setup_pendulum_model
from .utils import compute_lqr_gain_continuous_time

@dataclass
class ZanelliTighteningPendulumOcpOptions:
    N_horizon: int = 100
    N_exact: int = 100
    T_horizon: float = 1.0
    qp_solver: str = "PARTIAL_CONDENSING_HPIPM"


@dataclass
class PendulumParameters:
    xs = np.array([0.0, 0.0, 0.0, 0.0])
    us = np.array([0.0])

@dataclass
class PendulumMpcParameters:
    Q = np.diag([1e-1, 1e0, 1e-1, 2e-3])
    R = np.diag([5e-4])

def formulate_ocp_without_solver_opts(cost_module="CONL") -> AcadosOcp:

    # create ocp object to formulate the OCP
    ocp = AcadosOcp()

    # set model
    model = setup_pendulum_model()
    ocp.model = model

    # set cost
    mpc_params = PendulumMpcParameters()
    Q_mat = mpc_params.Q
    R_mat = mpc_params.R

    x = ocp.model.x
    u = ocp.model.u
    nx = x.rows()
    nu = u.rows()
    ny = nx + nu
    ny_e = nx

    cost_W = scipy.linalg.block_diag(Q_mat, R_mat)

    # compute terminal cost
    cost_W_e, _ = compute_lqr_gain_continuous_time(model, PendulumParameters(), mpc_params)
    cost_W_e = Q_mat


    # set cost module
    if cost_module == "LINEAR_LS":
        ocp.cost.cost_type = "LINEAR_LS"
        ocp.cost.cost_type_e = "LINEAR_LS"

        ocp.cost.Vx = np.zeros((ny, nx))
        ocp.cost.Vx[:nx, :nx] = np.eye(nx)

        Vu = np.zeros((ny, nu))
        Vu[4, 0] = 1.0
        ocp.cost.Vu = Vu

        ocp.cost.Vx_e = np.eye(nx)
        # ocp.solver_options.fixed_hess = 1

    elif cost_module == "NONLINEAR_LS":
        ocp.cost.cost_type = "NONLINEAR_LS"
        ocp.cost.cost_type_e = "NONLINEAR_LS"

        ocp.model.cost_y_expr = ca.vertcat(x, u)
        ocp.model.cost_y_expr_e = x
    elif cost_module == "CONL":
        ocp.cost.cost_type = "CONVEX_OVER_NONLINEAR"
        ocp.cost.cost_type_e = "CONVEX_OVER_NONLINEAR"

        ocp.model.cost_y_expr = ca.vertcat(x, u)
        ocp.model.cost_y_expr_e = x

        r = ca.SX.sym("r", ny)
        r_e = ca.SX.sym("r_e", ny_e)
        ocp.model.cost_r_in_psi_expr = r
        ocp.model.cost_r_in_psi_expr_e = r_e

        ocp.model.cost_psi_expr = 0.5 * (r.T @ cost_W @ r)
        ocp.model.cost_psi_expr_e = 0.5 * (r_e.T @ cost_W_e @ r_e)

    elif cost_module == "EXTERNAL":
        ocp.cost.cost_type = "EXTERNAL"
        ocp.cost.cost_type_e = "EXTERNAL"

        ocp.model.cost_expr_ext_cost = (
            0.5 * ca.vertcat(x, u).T @ cost_W @ ca.vertcat(x, u)
        )
        ocp.model.cost_expr_ext_cost_e = 0.5 * x.T @ cost_W_e @ x
    else:
        raise Exception(f"Unknown cost_module {cost_module}")

    if cost_module in ["LINEAR_LS", "NONLINEAR_LS", "CONL"]:
        ocp.cost.yref = np.zeros((ny,))
        ocp.cost.yref_e = np.zeros((ny_e,))
    if cost_module in ["LINEAR_LS", "NONLINEAR_LS"]:
        ocp.cost.W_e = cost_W_e
        ocp.cost.W = cost_W

    # set constraints
    Fmax = 12
    ocp.constraints.lbu = np.array([-Fmax])
    ocp.constraints.ubu = np.array([+Fmax])
    ocp.constraints.idxbu = np.array([0])

    ocp.constraints.x0 = np.array([0.0, np.pi, 0.0, 0.0])
    return ocp


# TODO: add this to AcadosOcp class?
def formulate_constraint_as_log_barrier(
        ocp: AcadosOcp,
        constr_expr: ca.SX,
        weight: float,
        upper_bound: Optional[float],
        lower_bound: Optional[float],
        residual_name: str = "new_residual",
        constraint_type: str = "path",
    ) -> None:
    """
    Formulate a constraint as an log-barrier term and add it to the current cost.
    """
    from scipy.linalg import block_diag
    import casadi as ca

    casadi_symbol = ocp.model.get_casadi_symbol()

    if upper_bound is None and lower_bound is None:
        raise ValueError("Either upper or lower bound must be provided.")

    # compute violation expression
    log_barrier_input = casadi_symbol("log_barrier_input", 0, 1)
    if upper_bound is not None:
        log_barrier_input = ca.vertcat(log_barrier_input, constr_expr - upper_bound)
    if lower_bound is not None:
        log_barrier_input = ca.vertcat(log_barrier_input, lower_bound - constr_expr)
    y_ref_new = np.zeros((log_barrier_input.shape[0],))


    # add penalty as cost
    if constraint_type == "path":
        ocp.cost.yref = np.concatenate((ocp.cost.yref, y_ref_new))
        ocp.model.cost_y_expr = ca.vertcat(ocp.model.cost_y_expr, log_barrier_input)
        if ocp.cost.cost_type == "CONVEX_OVER_NONLINEAR":
            new_residual = casadi_symbol(residual_name, log_barrier_input.shape)
            ocp.model.cost_r_in_psi_expr = ca.vertcat(ocp.model.cost_r_in_psi_expr, new_residual)
            for i in range(log_barrier_input.shape[0]):
                ocp.model.cost_psi_expr -= weight * ca.log(new_residual[i])
        # elif ocp.cost.cost_type == "EXTERNAL":
        #     ocp.model.cost_expr_ext_cost += .5 * weight * violation_expr**2
        else:
            raise NotImplementedError(f"formulate_constraint_as_L2_penalty not implemented for path cost with cost_type {ocp.cost.cost_type}.")
    elif constraint_type == "initial":
        raise NotImplementedError("TODO!")
    elif constraint_type == "terminal":
        raise NotImplementedError("TODO!")
    return ocp


def formulate_tightened_ocp_without_solver_opts():
    ocp = formulate_ocp_without_solver_opts(cost_module="CONL")
    constr_expr = ocp.model.u[ocp.constraints.idxbu]
    ocp = formulate_constraint_as_log_barrier(ocp, constr_expr, 5., ocp.constraints.ubu, ocp.constraints.lbu)
    ocp.constraints.lbu = np.array([])
    ocp.constraints.ubu = np.array([])
    ocp.constraints.idxbu = np.array([])
    return ocp

def formulate_zanelli_tightening_pendulum_ocp(options: ZanelliTighteningPendulumOcpOptions):

    if options.N_exact == options.N_horizon:
        ocp = formulate_ocp_without_solver_opts()
        ocp.solver_options.N_horizon = options.N_horizon
    elif options.N_exact < options.N_horizon:
        ocp = AcadosMultiphaseOcp(N_list=[options.N_exact, options.N_horizon - options.N_exact])
        ocp.set_phase(formulate_ocp_without_solver_opts(), 0)
        ocp.set_phase(formulate_tightened_ocp_without_solver_opts(), 1)
    # set options
    ocp.solver_options.qp_solver = options.qp_solver

    if options.qp_solver.startswith("PARTIAL_CONDENSING"):
        ocp.solver_options.qp_solver_cond_N = options.N_exact
        block_size = options.N_exact * [1]
        if options.N_horizon > options.N_exact:
            block_size += [options.N_horizon - options.N_exact]
        else:
            block_size += [0]
        ocp.solver_options.qp_solver_cond_block_size = block_size
    elif options.qp_solver in ["FULL_CONDENSING_QPOASES", "FULL_CONDENSING_DAQP"]:
        ocp.solver_options.qp_solver_iter_max = 200

    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"  # 'GAUSS_NEWTON', 'EXACT'
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.sim_method_num_stages = 4
    ocp.solver_options.nlp_solver_type = "SQP_RTI"
    ocp.solver_options.tf = options.T_horizon
    return ocp
