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

from acados_template import AcadosOcp, AcadosModel


def setup_pendulum_rate_model() -> AcadosModel:
    model_name = "pendulum"

    # constants
    M = 1.0  # mass of the cart [kg] -> now estimated
    m = 0.1  # mass of the ball [kg]
    g = 9.81  # gravity constant [m/s^2]
    l = 0.8  # length of the rod [m]

    # set up states & controls
    x1 = ca.SX.sym("x1")
    theta = ca.SX.sym("theta")
    v1 = ca.SX.sym("v1")
    dtheta = ca.SX.sym("dtheta")
    F = ca.SX.sym("F")

    x = ca.vertcat(x1, theta, v1, dtheta, F)

    dF = ca.SX.sym("dF")
    u = dF

    # xdot
    nx = x.rows()
    xdot = ca.SX.sym("xdot", nx, 1)

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
        dF,
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
    model.x_labels = ['$x$ [m]', r'$\theta$ [rad]', '$v$ [m]', r'$\dot{\theta}$ [rad/s]' , '$F$ [N]' ]
    model.u_labels = ['$dF$']
    model.t_label = '$t$ [s]'

    return model


@dataclass
class PendulumRateOcpOptions:
    qp_solver: str = "PARTIAL_CONDENSING_HPIPM"
    nlp_solver: str = "SQP"
    cost_variant: str = "NONLINEAR_LS"
    cost_discretization: str = "EULER"
    integrator_type: str = "ERK"
    sim_method_jac_reuse: bool = True
    sim_method_num_steps: int = 1
    N_horizon: int = 100
    T_horizon: float = 1.0


def formulate_pendulum_rate_ocp(options: PendulumRateOcpOptions):

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
    model = setup_pendulum_rate_model()
    ocp.model = model

    Tf = options.T_horizon
    # set dimensions
    ocp.solver_options.N_horizon = N_horizon

    # set cost
    Q_mat = 2 * np.diag([1e3, 1e3, 1e-2, 1e-2, 1e-2])
    # R_mat = 2 * np.diag([1e-3])

    x = ocp.model.x
    u = ocp.model.u
    nx = x.rows()
    nu = u.rows()
    ny_e = nx

    cost_W = scipy.linalg.block_diag(Q_mat)
    # cost_W = scipy.linalg.block_diag(Q_mat, R_mat)
    if cost_variant == "NONLINEAR_LS":
        ocp.cost.cost_type = "NONLINEAR_LS"
        ocp.cost.cost_type_e = "NONLINEAR_LS"

        # ocp.model.cost_y_expr = ca.vertcat(x, u)
        ocp.model.cost_y_expr = ca.vertcat(x)
        ocp.model.cost_y_expr_e = x
        ny = ocp.model.cost_y_expr.rows()

    if cost_variant in ["LINEAR_LS", "NONLINEAR_LS", "CONL", "CONL_LARGE"]:
        ocp.cost.yref = np.zeros((ny,))
        ocp.cost.yref_e = np.zeros((ny_e,))
    if cost_variant in ["LINEAR_LS", "NONLINEAR_LS"]:
        ocp.cost.W_e = Q_mat
        ocp.cost.W = cost_W

    # set constraints
    Fmax = 80
    ocp.constraints.lbx = np.array([-Fmax])
    ocp.constraints.ubx = np.array([+Fmax])
    ocp.constraints.idxbx = np.array([4])

    dFmax = 2000
    ocp.constraints.lbu = np.array([-dFmax])
    ocp.constraints.ubu = np.array([+dFmax])
    ocp.constraints.idxbu = np.array([0])

    ocp.constraints.x0 = np.array([0.0, np.pi/10, 0.0, 0.0, 0.0])

    # set options
    ocp.solver_options.qp_solver = qp_solver
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"  # 'GAUSS_NEWTON', 'EXACT'
    ocp.solver_options.integrator_type = integrator_type
    ocp.solver_options.cost_discretization = cost_discretization
    ocp.solver_options.sim_method_jac_reuse = sim_method_jac_reuse
    ocp.solver_options.sim_method_num_steps = options.sim_method_num_steps
    ocp.solver_options.nlp_solver_type = nlp_solver  # SQP_RTI, SQP
    ocp.solver_options.tf = Tf
    return ocp

