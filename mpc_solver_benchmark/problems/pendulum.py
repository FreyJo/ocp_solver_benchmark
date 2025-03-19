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

import numpy as np
import casadi as ca
import scipy

from dataclasses import dataclass

from acados_template import AcadosOcp, AcadosModel, AcadosOcpOptions, AcadosOcpSolver, ACADOS_INFTY

@dataclass
class PendulumModelParameters:
    # constants
    M : float = 1.0 # mass of the cart [kg]
    m : float = 0.1 # mass of the ball [kg]
    g : float = 9.81 # gravity constant [m/s^2]
    l : float = 0.8 # length of the rod [m]

def setup_pendulum_model() -> AcadosModel:
    model_name = "pendulum"

    params = PendulumModelParameters()

    # constants
    M = params.M
    m = params.m
    g = params.g
    l = params.l

    # set up states & controls
    x1 = ca.SX.sym("x1")
    theta = ca.SX.sym("theta")
    v1 = ca.SX.sym("v1")
    dtheta = ca.SX.sym("dtheta")

    x = ca.vertcat(x1, theta, v1, dtheta)

    F = ca.SX.sym("F")
    u = ca.vertcat(F)

    # xdot
    x1_dot = ca.SX.sym("x1_dot")
    theta_dot = ca.SX.sym("theta_dot")
    v1_dot = ca.SX.sym("v1_dot")
    dtheta_dot = ca.SX.sym("dtheta_dot")

    xdot = ca.vertcat(x1_dot, theta_dot, v1_dot, dtheta_dot)

    # common expression
    cos_theta = ca.cos(theta)
    sin_theta = ca.sin(theta)
    dtheta_squared = dtheta * dtheta
    denominator = M + m - m * cos_theta * cos_theta
    cos_theta_times_sin_theta = cos_theta * sin_theta

    # dynamics
    f_expl = ca.vertcat(
        v1,
        dtheta,
        (-m * l * sin_theta * dtheta_squared + m * g * cos_theta_times_sin_theta + F)
        / denominator,
        (
            -m * l * cos_theta_times_sin_theta * dtheta_squared
            + F * cos_theta
            + (M + m) * g * sin_theta
        )
        / (l * denominator),
    )

    f_impl = xdot - f_expl

    model = AcadosModel()

    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.x = x
    model.xdot = xdot
    model.u = u
    model.name = model_name

    # add meta information
    model.x_labels = [r'$p_{\mathrm{x}}$ [m]', r'$\theta$ [rad]', '$v$ [m]', r'$\dot{\theta}$ [rad/s]']
    model.u_labels = ['$F$ [N]']
    model.t_label = '$t$ [s]'

    return model

def setup_free_time_pendulum_ode_model() -> AcadosModel:

    model_name = 'free_time_pendulum'

    pars = PendulumModelParameters()

    # set up states & controls
    T = ca.SX.sym('T')
    x1      = ca.SX.sym('x1')
    theta   = ca.SX.sym('theta')
    v1      = ca.SX.sym('v1')
    dtheta  = ca.SX.sym('dtheta')

    x = ca.vertcat(T, x1, theta, v1, dtheta)

    F = ca.SX.sym('F')
    u = ca.vertcat(F)

    # xdot
    T_dot       = ca.SX.sym('T_dot')
    x1_dot      = ca.SX.sym('x1_dot')
    theta_dot   = ca.SX.sym('theta_dot')
    v1_dot      = ca.SX.sym('v1_dot')
    dtheta_dot  = ca.SX.sym('dtheta_dot')

    xdot = ca.vertcat(T_dot, x1_dot, theta_dot, v1_dot, dtheta_dot)

    # dynamics
    cos_theta = ca.cos(theta)
    sin_theta = ca.sin(theta)
    denominator = pars.M + pars.m - pars.m*cos_theta*cos_theta
    f_expl = ca.vertcat(0,
                        T*v1,
                        T*dtheta,
                        T*((-pars.m*pars.l*sin_theta*dtheta*dtheta + pars.m*pars.g*cos_theta*sin_theta+F)/denominator),
                        T*((-pars.m*pars.l*cos_theta*sin_theta*dtheta*dtheta + F*cos_theta+(pars.M+pars.m)*pars.g*sin_theta)/(pars.l*denominator))
                        )

    f_impl = xdot - f_expl

    model = AcadosModel()

    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.x = x
    model.xdot = xdot
    model.u = u
    model.name = model_name

    model.x_labels = ['$T$ [s]', r'$p_{\mathrm{x}}$ [m]', r'$\theta$ [rad]', '$v$ [m]', r'$\dot{\theta}$ [rad/s]']
    model.u_labels = ['$F$ [N]']
    model.t_label = '$t$ [s]'

    return model


@dataclass
class PendulumOcpOptions:
    qp_solver: str = "PARTIAL_CONDENSING_HPIPM"
    nlp_solver: str = "SQP"
    cost_variant: str = "LINEAR_LS"
    cost_discretization: str = "EULER"
    integrator_type: str = "ERK"
    sim_method_jac_reuse: bool = True
    sim_method_num_steps: int = 1
    sim_method_num_stages: int = 4
    N_horizon: int = 100
    T_horizon: float = 1.0


def formulate_pendulum_ocp(options: PendulumOcpOptions):

    # unpack options
    N_horizon = options.N_horizon
    qp_solver = options.qp_solver
    nlp_solver = options.nlp_solver
    cost_variant = options.cost_variant
    cost_discretization = options.cost_discretization
    integrator_type = options.integrator_type
    sim_method_jac_reuse = options.sim_method_jac_reuse

    # create ocp object to formulate the OCP
    ocp = AcadosOcp()

    # set model
    model = setup_pendulum_model()
    ocp.model = model

    Tf = options.T_horizon
    # set dimensions
    ocp.solver_options.N_horizon = N_horizon

    # set cost
    Q_mat = 2 * np.diag([1e3, 1e3, 1e-2, 1e-2])
    R_mat = 2 * np.diag([1e-2])

    x = ocp.model.x
    u = ocp.model.u
    nx = x.rows()
    nu = u.rows()
    ny = nx + nu
    ny_e = nx

    cost_W = scipy.linalg.block_diag(Q_mat, R_mat)
    if cost_variant == "LINEAR_LS":
        ocp.cost.cost_type = "LINEAR_LS"
        ocp.cost.cost_type_e = "LINEAR_LS"

        ocp.cost.Vx = np.zeros((ny, nx))
        ocp.cost.Vx[:nx, :nx] = np.eye(nx)

        Vu = np.zeros((ny, nu))
        Vu[4, 0] = 1.0
        ocp.cost.Vu = Vu

        ocp.cost.Vx_e = np.eye(nx)
        # ocp.solver_options.fixed_hess = 1

    elif cost_variant == "NONLINEAR_LS":
        ocp.cost.cost_type = "NONLINEAR_LS"
        ocp.cost.cost_type_e = "NONLINEAR_LS"

        ocp.model.cost_y_expr = ca.vertcat(x, u)
        ocp.model.cost_y_expr_e = x
    elif cost_variant == "CONL":
        ocp.cost.cost_type = "CONVEX_OVER_NONLINEAR"
        ocp.cost.cost_type_e = "CONVEX_OVER_NONLINEAR"

        ocp.model.cost_y_expr = ca.vertcat(x, u)
        ocp.model.cost_y_expr_e = x

        r = ca.SX.sym("r", ny)
        r_e = ca.SX.sym("r_e", ny_e)
        ocp.model.cost_r_in_psi_expr = r
        ocp.model.cost_r_in_psi_expr_e = r_e

        ocp.model.cost_psi_expr = 0.5 * (r.T @ cost_W @ r)
        ocp.model.cost_psi_expr_e = 0.5 * (r_e.T @ Q_mat @ r_e)
    elif cost_variant == "CONL_LARGE":
        ocp.cost.cost_type = "CONVEX_OVER_NONLINEAR"
        ocp.cost.cost_type_e = "CONVEX_OVER_NONLINEAR"

        ocp.model.cost_y_expr = ca.vertcat(x, u, x, u)
        ocp.model.cost_y_expr_e = x
        ny *= 2

        r = ca.SX.sym("r", ny)
        r_e = ca.SX.sym("r_e", ny_e)
        ocp.model.cost_r_in_psi_expr = r
        ocp.model.cost_r_in_psi_expr_e = r_e

        ocp.model.cost_psi_expr = 0.5 * (r.T @ scipy.linalg.block_diag(cost_W, cost_W) @ r)
        ocp.model.cost_psi_expr_e = 0.5 * (r_e.T @ Q_mat @ r_e)
    elif cost_variant == "EXTERNAL":
        ocp.cost.cost_type = "EXTERNAL"
        ocp.cost.cost_type_e = "EXTERNAL"

        ocp.model.cost_expr_ext_cost = (
            0.5 * ca.vertcat(x, u).T @ cost_W @ ca.vertcat(x, u)
        )
        ocp.model.cost_expr_ext_cost_e = 0.5 * x.T @ Q_mat @ x
    else:
        raise Exception("Unknown cost_variant.")

    if cost_variant in ["LINEAR_LS", "NONLINEAR_LS", "CONL", "CONL_LARGE"]:
        ocp.cost.yref = np.zeros((ny,))
        ocp.cost.yref_e = np.zeros((ny_e,))
    if cost_variant in ["LINEAR_LS", "NONLINEAR_LS"]:
        ocp.cost.W_e = Q_mat
        ocp.cost.W = cost_W

    # set constraints
    Fmax = 80
    ocp.constraints.lbu = np.array([-Fmax])
    ocp.constraints.ubu = np.array([+Fmax])
    ocp.constraints.idxbu = np.array([0])

    ocp.constraints.x0 = np.array([0.0, np.pi/10, 0.0, 0.0])

    # set options
    ocp.solver_options.qp_solver = qp_solver
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"  # 'GAUSS_NEWTON', 'EXACT'
    ocp.solver_options.integrator_type = integrator_type
    ocp.solver_options.cost_discretization = cost_discretization
    ocp.solver_options.sim_method_jac_reuse = sim_method_jac_reuse
    ocp.solver_options.sim_method_num_steps = options.sim_method_num_steps
    ocp.solver_options.sim_method_num_stages = options.sim_method_num_stages
    ocp.solver_options.nlp_solver_type = nlp_solver  # SQP_RTI, SQP
    ocp.solver_options.tf = Tf
    ocp.solver_options.log_primal_step_norm = True
    return ocp


@dataclass
class PendulumParmeters:
    N : int = 100
    nx : int = 5
    nu : int = 1
    Tf : float = 1.0

    # Parameters
    max_f : float = 5.
    max_x1 : float = 1.0
    max_v : float = 2.0

    x1_0 : float = 0.0
    theta_0 : float = np.pi
    dx1_0 : float = 0.0
    dtheta_0 : float = 0.0

    theta_f : float = 0.0
    dx1_f : float = 0.0
    dtheta_f : float = 0.0

def formulate_time_optimal_swing_up(options: AcadosOcpOptions):
    # create ocp object to formulate the OCP
    params = PendulumParmeters()
    ocp = AcadosOcp()

    # set model
    model = setup_free_time_pendulum_ode_model()
    ocp.model = model

    ###########################################################################
    # Define constraints
    ###########################################################################

    # Initial conditions
    ocp.constraints.lbx_0 = np.array([0.0, params.x1_0, params.theta_0, params.dx1_0, params.dtheta_0])
    ocp.constraints.ubx_0 = np.array([ACADOS_INFTY, params.x1_0, params.theta_0, params.dx1_0, params.dtheta_0])
    ocp.constraints.idxbx_0 = np.array([0, 1, 2, 3, 4])

    # Actuator constraints
    ocp.constraints.lbu = np.array([-params.max_f])
    ocp.constraints.ubu = np.array([+params.max_f])
    ocp.constraints.idxbu = np.array([0])

    ocp.constraints.lbx = np.array([0.0, -params.max_x1, -params.max_v])
    ocp.constraints.ubx = np.array([ACADOS_INFTY, params.max_x1, params.max_v])
    ocp.constraints.idxbx = np.array([0, 1, 3])

    # Terminal constraints
    ocp.constraints.lbx_e = np.array([0.0, params.theta_f, params.dx1_f, params.dtheta_f])
    ocp.constraints.ubx_e = np.array([ACADOS_INFTY, params.theta_f, params.dx1_f, params.dtheta_f])
    ocp.constraints.idxbx_e = np.array([0, 2, 3, 4])

    ###########################################################################
    # Define objective function
    ###########################################################################
    ocp.cost.cost_type_e = 'EXTERNAL'
    ocp.model.cost_expr_ext_cost_e = model.x[0]

    # Setup options
    ocp.solver_options = options
    ocp.solver_options.N_horizon = params.N
    ocp.solver_options.tf = params.Tf
    ocp.solver_options.integrator_type = 'ERK'

    return ocp

def initialize_time_optimal_swing_up(ocp_solver: AcadosOcpSolver):
    # Initial guess
    T0 = 1.0
    params = PendulumParmeters()

    for i in range(params.N):
        ocp_solver.set(i, "x", np.array([T0, 0.0, np.pi, 0.0, 0.0]))
    ocp_solver.set(params.N, "x", np.array([T0, 0.0, np.pi, 0.0, 0.0]))