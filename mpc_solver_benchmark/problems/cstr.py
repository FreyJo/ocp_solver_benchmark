import numpy as np
import casadi as ca
from dataclasses import dataclass
from acados_template import AcadosModel, AcadosOcp, AcadosMultiphaseOcp
from typing import Optional, Union, Tuple
from ..reference import Reference
from .utils import reformulate_as_clc_ocp, compute_lqr_gain_continuous_time

@dataclass
class CstrParameters:
    # nominal parameter values
    F0: float = 0.1  # m^3/min
    T0: float = 350.0  # K
    c0: float = 1.0  # kmol/m^3
    r: float = 0.219  # m
    k0: float = 7.2 * 1e10  # 1/min
    EbR: float = 8750  # K
    U: float = 54.94  # kJ / (min*m^2*K)
    rho: float = 1000  # kg / m^3
    Cp: float = 0.239  # kJ / (kg*K)
    dH: float = -5 * 1e4  # kJ / kmol
    # to avoid division by zero
    eps: float = 1e-5  # m
    xs = np.array([0.878, 324.5, 0.659])
    us = np.array([300, 0.1])


def setup_cstr_model(params: CstrParameters) -> AcadosModel:

    model_name = "cstr_ode"

    # set up states
    c = ca.SX.sym("c")  # molar concentration of species A
    T = ca.SX.sym("T")  # reactor temperature
    h = ca.SX.sym("h")  # level of the tank

    x = ca.vertcat(c, T, h)

    # controls
    Tc = ca.SX.sym("Tc")  # temperature of coolant liquid
    F = ca.SX.sym("F")  # outlet flowrate

    u = ca.vertcat(Tc, F)

    # xdot
    c_dot = ca.SX.sym("c_dot")
    T_dot = ca.SX.sym("T_dot")
    h_dot = ca.SX.sym("h_dot")

    xdot = ca.vertcat(c_dot, T_dot, h_dot)

    # dynamics
    A_const = np.pi * params.r**2
    denom = A_const * (h + params.eps)
    k = params.k0 * ca.exp(-params.EbR / T)
    rate = k * c

    f_expl = ca.vertcat(
        params.F0 * (params.c0 - c) / denom - rate,
        params.F0 * (params.T0 - T) / denom
        - params.dH / (params.rho * params.Cp) * rate
        + 2 * params.U / (params.r * params.rho * params.Cp) * (Tc - T),
        (params.F0 - F) / A_const,
    )

    f_impl = xdot - f_expl

    model = AcadosModel()

    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.x = x
    model.xdot = xdot
    model.u = u
    model.t = ca.SX.sym("t")
    model.name = model_name

    # add meta information
    model.x_labels = ["$c$ [kmol/m$^3$]", "$T$ [K]", "$h$ [m]"]
    model.u_labels = ["$T_c$ [K]", "$F$ [m$^3$/min]"]
    model.t_label = "$t$ [min]"

    return model


@dataclass
class CstrOcpOptions:
    qp_solver: str = "PARTIAL_CONDENSING_HPIPM"
    # qp_solver: str = "FULL_CONDENSING_DAQP"
    nlp_solver: str = "SQP_RTI"
    cost_discretization: str = "INTEGRATOR"
    dt0: float = 0.1
    N_horizon: int = 16
    T_horizon: float = 5.0
    sim_method_num_steps: int = 100
    sim_method_num_stages: int = 4
    sim_method_newton_iter: int = 7
    known_reference: bool = True
    clc_horizon: int = 0


class MpcCstrParameters:
    def __init__(self, xs: np.ndarray, us: np.ndarray, dt: float = 0.25, linear_mpc: bool = False):
        self.umin = np.array([0.95, 0.85]) * us
        self.umax = np.array([1.05, 1.15]) * us
        self.Q = np.diag(1.0 / xs**2)
        self.R = np.diag(1.0 / us**2)
        self.dt = dt
        self.linear_mpc = linear_mpc
        # NOTE: computed with compute_lqr_gain() from cstr_utils.py
        self.P = np.array([
            [5.92981953e-01, -8.40033347e-04, -1.54536980e-02],
            [-8.40033347e-04, 7.75225208e-06, 2.30677411e-05],
            [-1.54536980e-02, 2.30677411e-05, 2.59450075e00],
        ])


def formulate_cstr_ocp(ocp_options: CstrOcpOptions, reference: Optional[Reference] = None) -> AcadosOcp:

    cstr_params = CstrParameters()
    model = setup_cstr_model(cstr_params)
    mpc_params = MpcCstrParameters(cstr_params.xs, cstr_params.us)

    if ocp_options.clc_horizon == 0:
        ocp = formulate_cstr_ocp_single_phase(model, cstr_params, mpc_params, ocp_options, reference)
    else:
        N_horizon_0 = ocp_options.N_horizon - ocp_options.clc_horizon
        ocp = AcadosMultiphaseOcp([N_horizon_0, ocp_options.clc_horizon])

        ocp_0 = formulate_cstr_ocp_single_phase(model, cstr_params, mpc_params, ocp_options, reference)

        model = setup_cstr_model(cstr_params)
        ocp_1 = formulate_cstr_ocp_single_phase(model, cstr_params, mpc_params, ocp_options, reference)
        _, K = compute_lqr_gain_continuous_time(ocp_1.model, cstr_params, mpc_params)
        if reference is None:
            x_ref = cstr_params.xs
            u_ref = cstr_params.us
        else:
            nx = cstr_params.xs.shape[0]
            y_ref = reference.get_casadi_expression(ocp_1.model.t0)
            x_ref = y_ref[:nx]
            u_ref = y_ref[nx:]
        ocp_1 = reformulate_as_clc_ocp(ocp_1, K, x_ref, u_ref)

        ocp.set_phase(ocp_0, 0)
        ocp.set_phase(ocp_1, 1)

        ocp.solver_options = ocp_0.solver_options

    return ocp


def formulate_cstr_ocp_single_phase(model: AcadosModel, cstr_params: CstrParameters, mpc_params: MpcCstrParameters, ocp_options: CstrOcpOptions, reference: Optional[Reference] = None):
    ocp = AcadosOcp()

    # set model
    ocp.model = model
    x = model.x
    u = model.u
    nx = x.shape[0]
    nu = u.shape[0]

    # augment model with t0. This is necessary for time-varying references.
    ocp.augment_with_t0_param()
    t0 = ocp.model.t0

    # set prediction horizon
    ocp.solver_options.N_horizon = ocp_options.N_horizon
    ocp.solver_options.tf = ocp_options.T_horizon
    ocp.solver_options.time_steps = np.array([ocp_options.dt0] + (ocp_options.N_horizon-1) * [(ocp_options.T_horizon - ocp_options.dt0) / (ocp_options.N_horizon-1)])

    # set cost
    ocp.cost.W_e = mpc_params.P
    ocp.cost.W = ca.diagcat(mpc_params.Q, mpc_params.R).full()

    ocp.cost.cost_type = "NONLINEAR_LS"
    ocp.cost.cost_type_e = "NONLINEAR_LS"

    ocp.model.cost_y_expr = ca.vertcat(x, u)
    ocp.model.cost_y_expr_e = x

    if reference is not None and ocp_options.known_reference and ocp_options.cost_discretization == "INTEGRATOR":
        ref_expression = reference.get_casadi_expression(model.t + t0)
        ocp.model.cost_y_expr -= ref_expression
        ocp.model.cost_y_expr_e -= ref_expression[:nx]

    ocp.cost.yref = np.zeros((nx + nu,))
    ocp.cost.yref_e = np.zeros((nx,))

    # set constraints
    ocp.constraints.lbu = mpc_params.umin
    ocp.constraints.ubu = mpc_params.umax
    ocp.constraints.idxbu = np.arange(nu)

    ocp.constraints.x0 = cstr_params.xs

    # set options
    ocp.solver_options.qp_solver = ocp_options.qp_solver
    ocp.solver_options.qp_solver_cond_N = ocp_options.N_horizon
    ocp.solver_options.cost_discretization = ocp_options.cost_discretization

    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.nlp_solver_type = ocp_options.nlp_solver

    ocp.solver_options.integrator_type = "IRK"
    ocp.solver_options.sim_method_num_stages = ocp_options.sim_method_num_stages
    ocp.solver_options.sim_method_num_steps = ocp_options.sim_method_num_steps
    ocp.solver_options.sim_method_newton_iter = ocp_options.sim_method_newton_iter

    ocp.solver_options.levenberg_marquardt = 1e-5
    ocp.solver_options.line_search_use_sufficient_descent

    return ocp
