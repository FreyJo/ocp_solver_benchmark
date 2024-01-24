import numpy as np
import casadi as ca
import scipy.linalg

from dataclasses import dataclass

from acados_template import AcadosOcp, AcadosModel, AcadosMultiphaseOcp

from typing import Optional

X0 = np.array([1.0, 1.0, 0.0, np.pi, 0.0])  # Intital state
a_max = 500  # Define the max force allowed
T_max = 20.
omega_max = .5

def setup_diff_drive_model() -> AcadosModel:
    model_name = "diff_drive"

    # set up states & controls
    x_pos = ca.SX.sym("x_pos")
    y_pos = ca.SX.sym("y_pos")
    v = ca.SX.sym("v")
    theta = ca.SX.sym("theta")
    omega = ca.SX.sym("omega")

    x = ca.vertcat(x_pos, y_pos, v, theta, omega)
    nx = x.rows()

    tau_r = ca.SX.sym("tau_r")
    tau_l = ca.SX.sym("tau_l")
    u = ca.vertcat(tau_r, tau_l)

    # xdot
    xdot = ca.SX.sym("xdot", nx)

    # dynamics
    m = 220
    mc = 200
    L = 0.32
    R = 0.16
    d = 0.01
    I = 9.6
    Iw = 0.1
    term = (tau_r + tau_l)/R
    term_2 = 2*Iw/R**2
    v_dot = (term + mc * d* omega**2)/(m + term_2)
    omega_dot = (L*(tau_r - tau_l)/R - mc*d*omega*v)/(I + L**2*term_2)
    f_expl = ca.vertcat(v * ca.cos(theta), v * ca.sin(theta), v_dot, omega, omega_dot)

    f_impl = xdot - f_expl

    model = AcadosModel()

    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.x = x
    model.xdot = xdot
    model.u = u
    model.name = model_name

    model.t_label = "$t$ [s]"
    model.x_labels = ["$x$", "$y$", "$v$", "$\\theta$", "$\\omega$"]
    model.u_labels = [r"$\tau_r$", r"$\tau_l$"]

    return model


def setup_unicycle_model_actuators() -> AcadosModel:
    model_name = "actuators_diff_drive"

    # set up states & controls
    x_pos = ca.SX.sym("x_pos")
    y_pos = ca.SX.sym("y_pos")
    v = ca.SX.sym("v")
    theta = ca.SX.sym("theta")
    omega = ca.SX.sym("omega")
    x = ca.vertcat(x_pos, y_pos, v, theta, omega)

    tau_r = ca.SX.sym("tau_r")
    tau_l = ca.SX.sym("tau_l")
    # u = ca.vertcat(tau_r, tau_l)

    # dynamics
    m = 220
    mc = 200
    L = 0.32
    R = 0.16
    d = 0.01
    I = 9.6
    Iw = 0.1
    term = (tau_r + tau_l)/R
    term_2 = 2*Iw/R**2
    v_dot = (term + mc * d* omega**2)/(m + term_2)
    omega_dot = (L*(tau_r - tau_l)/R - mc*d*omega*v)/(I + L**2*term_2)
    f_expl = ca.vertcat(v * ca.cos(theta), v * ca.sin(theta), v_dot, omega, omega_dot)

    # actuators
    #  voltage
    V_r = ca.SX.sym("V_r")
    V_l = ca.SX.sym("V_l")

    #  current
    i_r = ca.SX.sym("i_r")
    i_l = ca.SX.sym("i_l")
    K1 = 1.0  # motor constants
    K2 = 1.0
    L_inductance = 0.0001 # coil inductance
    R_resistance = 0.05 # coil resistance

    # append to ODE
    phi1_dot = (f_expl[0] * ca.cos(theta) + f_expl[1] * ca.sin(theta) + L * omega) / R
    phi2_dot = (f_expl[0] * ca.cos(theta) + f_expl[1] * ca.sin(theta) - L * omega) / R
    i_r_dot = (- K1 * phi1_dot - R_resistance * i_r + V_r) / L_inductance
    i_l_dot = (- K2 * phi2_dot - R_resistance * i_l + V_l) / L_inductance

    f_expl = ca.vertcat(f_expl, i_r_dot, i_l_dot)
    x = ca.vertcat(x, i_r, i_l)

    # substitue
    f_expl = ca.substitute(f_expl, tau_r, K1 * i_r)
    f_expl = ca.substitute(f_expl, tau_l, K2 * i_l)
    u = ca.vertcat(V_r, V_l)

    # xdot
    nx = x.rows()
    xdot = ca.SX.sym("xdot", nx)
    f_impl = xdot - f_expl

    model = AcadosModel()

    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.x = x
    model.xdot = xdot
    model.u = u
    model.name = model_name

    model.t_label = "$t$ [s]"
    model.x_labels = ["$x$", "$y$", "$v$", "$\\theta$", "$\\omega$", "$i_r$", "$i_l$"]
    model.u_labels = [r"$V_r$", r"$V_l$"]

    return model


def setup_transition_model_actuators_to_diff_drive() -> AcadosModel:
    x_pos = ca.SX.sym("x_pos")
    y_pos = ca.SX.sym("y_pos")
    v = ca.SX.sym("v")
    theta = ca.SX.sym("theta")
    omega = ca.SX.sym("omega")
    i_1 = ca.SX.sym("i_1")
    i_2 = ca.SX.sym("i_2")

    x = ca.vertcat(x_pos, y_pos, v, theta, omega, i_1, i_2)

    model = AcadosModel()
    model.x = x
    model.name = "transition_actuators_to_diff_drive"
    model.disc_dyn_expr = ca.vertcat(x_pos, y_pos, v, theta, omega)
    return model


def setup_unicycle_model() -> AcadosModel:
    model_name = "unicycle"

    # set up states & controls
    x_pos = ca.SX.sym("x_pos")
    y_pos = ca.SX.sym("y_pos")
    v = ca.SX.sym("v")
    theta = ca.SX.sym("theta")
    omega = ca.SX.sym("omega")

    x = ca.vertcat(x_pos, y_pos, v, theta, omega)
    nx = x.rows()

    a = ca.SX.sym("a")
    T = ca.SX.sym("T")
    u = ca.vertcat(a, T)

    # xdot
    xdot = ca.SX.sym("xdot", nx)

    # dynamics
    mass = 200
    f_expl = ca.vertcat(v * ca.cos(theta), v * ca.sin(theta), a/mass, omega, T)

    f_impl = xdot - f_expl

    model = AcadosModel()

    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.x = x
    model.xdot = xdot
    model.u = u
    model.name = model_name

    model.t_label = "$t$ [s]"
    model.x_labels = ["$x$", "$y$", "$v$", "$\\theta$", "$\\dot{\\theta}$"]
    model.u_labels = ["$a$ [m/s$^2$]", "$T$"]

    return model


def setup_transition_model_diff_drive_to_simplified() -> AcadosModel:
    x_pos = ca.SX.sym("x_pos")
    y_pos = ca.SX.sym("y_pos")
    v = ca.SX.sym("v")
    theta = ca.SX.sym("theta")
    omega = ca.SX.sym("omega")

    x = ca.vertcat(x_pos, y_pos, v, theta, omega)

    model = AcadosModel()
    model.x = x
    model.name = "transition_diff_drive_to_simplified"
    model.disc_dyn_expr = ca.vertcat(x_pos, y_pos, v, theta)
    return model


def setup_simplified_unicycle_model() -> AcadosModel:
    model_name = "simple_unicycle"

    # set up states & controls
    x_pos = ca.SX.sym("x_pos")
    y_pos = ca.SX.sym("y_pos")
    v = ca.SX.sym("v")
    theta = ca.SX.sym("theta")

    x = ca.vertcat(x_pos, y_pos, v, theta)

    a = ca.SX.sym("a")
    omega = ca.SX.sym("omega")
    u = ca.vertcat(a, omega)

    # xdot
    x_dot = ca.SX.sym("x_dot")
    y_dot = ca.SX.sym("y_dot")
    v_dot = ca.SX.sym("v_dot")
    theta_dot = ca.SX.sym("theta_dot")

    xdot = ca.vertcat(x_dot, y_dot, v_dot, theta_dot)


    # dynamics
    mass = 200
    f_expl = ca.vertcat(v * ca.cos(theta),
                        v * ca.sin(theta),
                        a/mass,
                        omega)

    f_impl = xdot - f_expl

    model = AcadosModel()

    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.x = x
    model.xdot = xdot
    model.u = u
    model.name = model_name

    model.t_label = "$t$ [s]"
    model.x_labels = ["$x$", "$y$", "$v$", "$\\theta$"]
    model.u_labels = ["$a$ [m/s$^2$]", "$\\omega$ [rad/s]"]

    return model

@dataclass
class UnicycleOcpOptions:
    # qp_solver: str = "FULL_CONDENSING_DAQP"
    qp_solver: str = "PARTIAL_CONDENSING_HPIPM"
    N_horizon: int = 100
    T_horizon: float = 5
    sim_method_num_steps: int = 1
    cost_variant: str = "HUBER"
    cost_discretization: str = "INTEGRATOR"
    model_type: str = "diff_drive"
    globalization: str = "MERIT_BACKTRACKING"
    nlp_solver_type: str = "SQP"
    N_horizon_0: Optional[int] = None
    qp_solver_cond_N: int = 10


def unicycle_get_circular_constraints() -> list:
    # in format (x, y, r)
    # return [(0.3, 0.7, 0.2)]
    return []


def formulate_unicycle_ocp(options: UnicycleOcpOptions) -> AcadosOcp:
    if options.model_type == "combined":
        N_horizon_0 = options.N_horizon_0
        N_horizon_1 = options.N_horizon - N_horizon_0 - 1
        ocp = AcadosMultiphaseOcp([N_horizon_0, 1, N_horizon_1])
        ocp_0 = formulate_single_phase_unicycle_ocp(options, "diff_drive")
        ocp.set_phase(ocp_0, 0)

        # transition ocp.
        model = setup_transition_model_diff_drive_to_simplified()
        transition_ocp = AcadosOcp()
        transition_ocp.model = model
        transition_ocp.cost.cost_type = "NONLINEAR_LS"
        transition_ocp.model.cost_y_expr = model.x
        transition_ocp.cost.W = 1e-3 * np.eye(model.x.rows())
        transition_ocp.cost.yref = np.zeros((model.x.rows(),))

        ocp.set_phase(transition_ocp, 1)

        ocp_1 = formulate_single_phase_unicycle_ocp(options, "simplified_unicycle")
        ocp.set_phase(ocp_1, 2)
        ocp.solver_options = ocp_0.solver_options
        ocp.solver_options.tf = options.T_horizon
        dt = options.T_horizon / options.N_horizon
        ocp.solver_options.time_steps = np.array(N_horizon_0 * [dt] + [dt] + N_horizon_1 * [dt])

        ocp.mocp_opts.cost_discretization = [options.cost_discretization, "EULER", options.cost_discretization]
        ocp.mocp_opts.integrator_type = ["IRK", "DISCRETE", "IRK"]

    elif options.model_type == "actuators_to_diff_drive":
        N_horizon_0 = options.N_horizon_0
        N_horizon_1 = options.N_horizon - N_horizon_0
        dt = options.T_horizon / options.N_horizon

        ocp = AcadosMultiphaseOcp([N_horizon_0, 1, N_horizon_1])
        ocp_0 = formulate_diff_drive_actuators_ocp(options)
        ocp.set_phase(ocp_0, 0)

        # transition ocp.
        transition_ocp = AcadosOcp()
        model = setup_transition_model_actuators_to_diff_drive()
        transition_ocp.model = model
        transition_ocp.cost.cost_type = "NONLINEAR_LS"
        transition_ocp.model.cost_y_expr = model.x
        transition_ocp.cost.W = 1e-7 * np.eye(model.x.rows())
        transition_ocp.cost.W[-2:, -2:] = dt * 1e0 * np.eye(2)
        transition_ocp.cost.yref = np.zeros((model.x.rows(),))
        ocp.set_phase(transition_ocp, 1)

        ocp_1 = formulate_single_phase_unicycle_ocp(options, "diff_drive")
        ocp.set_phase(ocp_1, 2)
        ocp.solver_options = ocp_0.solver_options
        ocp.solver_options.tf = options.T_horizon + 1
        ocp.solver_options.time_steps = np.array(N_horizon_0 * [dt] + [1] + N_horizon_1 * [dt])

        ocp.mocp_opts.cost_discretization = [options.cost_discretization, "EULER", options.cost_discretization]
        ocp.mocp_opts.integrator_type = ["IRK", "DISCRETE", "IRK"]
    elif options.model_type == "actuators":
        ocp = formulate_diff_drive_actuators_ocp(options)
    else:
        ocp = formulate_single_phase_unicycle_ocp(options, options.model_type)
    return ocp


def set_solver_options_in_ocp(ocp: AcadosOcp, options: UnicycleOcpOptions):
    # set options
    ocp.solver_options.qp_solver = options.qp_solver
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.integrator_type = "IRK"
    ocp.solver_options.nlp_solver_type = options.nlp_solver_type
    ocp.solver_options.sim_method_num_stages = 3
    ocp.solver_options.sim_method_num_steps = options.sim_method_num_steps
    ocp.solver_options.globalization = options.globalization
    ocp.solver_options.cost_discretization = options.cost_discretization
    # ocp.solver_options.qp_solver_iter_max = 400
    ocp.solver_options.nlp_solver_max_iter = 400
    # ocp.solver_options.levenberg_marquardt = 1e-4
    # if options.nlp_solver_type == "SQP":
        # ocp.solver_options.globalization = "MERIT_BACKTRACKING"
    # ocp.solver_options.qp_solver_cond_N = options.qp_solver_cond_N
    ocp.solver_options.tol = 1e-6
    ocp.solver_options.qp_tol = 1e-2 * ocp.solver_options.tol
    # ocp.solver_options.nlp_solver_ext_qp_res = 1

def formulate_single_phase_unicycle_ocp(options: UnicycleOcpOptions, model_type) -> AcadosOcp:
    N_horizon = options.N_horizon

    # create ocp object to formulate the OCP
    ocp = AcadosOcp()

    if model_type == "unicycle":
        model = setup_unicycle_model()
    elif model_type == "simplified_unicycle":
        model = setup_simplified_unicycle_model()
    elif model_type == "diff_drive":
        model = setup_diff_drive_model()
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    ocp.model = model
    nx = model.x.rows()
    nu = model.u.rows()

    # set dimensions
    ocp.dims.N = N_horizon
    ocp.solver_options.tf = options.T_horizon

    # set cost
    if options.cost_variant == "LINEAR_LS":
        Q_mat = 2 * np.diag([1e3, 1e3, 1e-4, 1e-0, 1e-3])  # [x,y,x_d,y_d,th,th_d]
        R_mat = 2 * 5 * np.diag([1e-1, 1e-1])

        ocp.cost.cost_type = "LINEAR_LS"
        ocp.cost.cost_type_e = "LINEAR_LS"

        ny = nx + nu
        ny_e = nx

        ocp.cost.W_e = Q_mat
        ocp.cost.W = scipy.linalg.block_diag(Q_mat, R_mat)

        ocp.cost.Vx = np.zeros((ny, nx))
        ocp.cost.Vx[:nx, :nx] = np.eye(nx)

        Vu = np.zeros((ny, nu))
        Vu[nx : nx + nu, 0:nu] = np.eye(nu)
        ocp.cost.Vu = Vu

        ocp.cost.Vx_e = np.eye(nx)

        ocp.cost.yref = np.zeros((ny,))
        ocp.cost.yref_e = np.zeros((ny_e,))
    elif options.cost_variant == "HUBER":
        from acados_template import huber_loss
        ocp.cost.cost_type = "CONVEX_OVER_NONLINEAR"
        ocp.cost.cost_type_e = "CONVEX_OVER_NONLINEAR"

        r = ca.SX.sym("r", nx+nu)
        huber_delta = 1.0
        huber_tau = 0.1
        h_x = 5e2 * huber_loss(r[0], huber_delta, huber_tau)[0]
        h_y = 5e2 * huber_loss(r[1], huber_delta, huber_tau)[0]
        W_no_pos = 1e-4 * ca.DM.eye(nx-2+nu)
        if options.model_type == "diff_drive":
            W_no_pos[-2:, -2:] = 1e-4 @ ca.DM.eye(2)
        else:
            W_no_pos[-2:, -2:] = ca.diag([1e-5, 1e-4])

        ocp.model.cost_psi_expr = h_x + h_y + r[2:].T @ W_no_pos @ r[2:]
        ocp.model.cost_r_in_psi_expr = r
        ocp.model.cost_y_expr = ca.vertcat(model.x, model.u)

        W_no_pos_e = 1e-4 * ca.DM.eye(nx-2)
        W_no_pos_e[0, 0] = 1e3

        ocp.model.cost_psi_expr_e = 10*(h_x + h_y) + r[2:nx].T @ W_no_pos_e @ r[2:nx]
        ocp.model.cost_r_in_psi_expr_e = r[:nx]
        ocp.model.cost_y_expr_e = model.x

        ocp.cost.yref = np.zeros((nx+nu,))
        ocp.cost.yref_e = np.zeros((nx,))

    # set constraints
    ocp.constraints.idxbu = np.arange(nu)
    if model_type == "unicycle":
        ocp.constraints.lbu = np.array([-a_max, -T_max])
        ocp.constraints.ubu = np.array([+a_max, T_max])
        ocp.constraints.idxbx = np.array([2, 4])
        ocp.constraints.ubx = np.array([1., omega_max])
        ocp.constraints.lbx = np.array([0., -omega_max])
    elif model_type == "simplified_unicycle":
        ocp.constraints.ubu = np.array([a_max, omega_max])
        ocp.constraints.lbu = -ocp.constraints.ubu
        ocp.constraints.idxbx = np.array([2])
        ocp.constraints.ubx = np.array([1.,])
        ocp.constraints.lbx = np.array([0.,])
    elif model_type == "diff_drive":
        ocp.constraints.ubu = np.array([60, 60])
        ocp.constraints.lbu = -ocp.constraints.ubu
        ocp.constraints.idxbx = np.array([2, 4])
        ocp.constraints.ubx = np.array([1., omega_max])
        ocp.constraints.lbx = np.array([0., -omega_max])
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # # bound on v.
    # ocp.constraints.idxbx = np.array([2])
    # ocp.constraints.lbx = np.array([0.0])
    # ocp.constraints.ubx = np.array([4.0])

    # add obstacle
    circular_constraints = unicycle_get_circular_constraints()
    for x, y, r in circular_constraints:
        dist = ca.sqrt((model.x[0] - x) ** 2 + (model.x[1] - y) ** 2 + 1e-3)
        ocp.formulate_constraint_as_L2_penalty(dist, 1e6, lower_bound=r, upper_bound=None)

    ocp.constraints.x0 = X0[:nx]

    set_solver_options_in_ocp(ocp, options)
    return ocp



def formulate_diff_drive_actuators_ocp(options: UnicycleOcpOptions) -> AcadosOcp:
    N_horizon = options.N_horizon

    # create ocp object to formulate the OCP
    ocp = AcadosOcp()

    model_type = "actuators"
    if model_type == "actuators":
        model = setup_unicycle_model_actuators()
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    ocp.model = model
    nx = model.x.rows()
    nu = model.u.rows()

    # set dimensions
    ocp.dims.N = N_horizon
    ocp.solver_options.tf = options.T_horizon

    # set cost
    if options.cost_variant == "LINEAR_LS":
        Q_mat = 2 * np.diag([1e3, 1e3, 1e-4, 1e0, 1e-3, 5e-1, 5e-1])  # [x_pos, y_pos, v, theta, omega, i_1, i_2]
        # R_mat = 2 * 5 * np.diag([1e-5, 1e-5])

        ocp.cost.cost_type = "LINEAR_LS"
        ocp.cost.cost_type_e = "LINEAR_LS"

        ny = nx
        ny_e = nx

        ocp.cost.W_e = Q_mat
        ocp.cost.W = Q_mat

        ocp.cost.Vx = np.zeros((ny, nx))
        ocp.cost.Vx[:nx, :nx] = np.eye(nx)

        Vu = np.zeros((ny, nu))
        # Vu[nx : nx + nu, 0:nu] = np.eye(nu)
        ocp.cost.Vu = Vu

        ocp.cost.Vx_e = np.eye(nx)

        ocp.cost.yref = np.zeros((ny,))
        ocp.cost.yref_e = np.zeros((ny_e,))
    else:
        raise ValueError(f"Unknown cost variant: {options.cost_variant}")

    # set constraints
    ocp.constraints.idxbu = np.arange(nu)
    ocp.constraints.ubu = np.array([10.0, 10.0])
    ocp.constraints.lbu = -ocp.constraints.ubu

    # bound on v, omega
    ocp.constraints.idxbx = np.array([2, 4])
    ocp.constraints.ubx = np.array([1., omega_max])
    ocp.constraints.lbx = np.array([0., -omega_max])

    # compute power
    ocp.model.con_h_expr = ca.vertcat(model.u[0] * model.x[5], model.u[1] * model.x[6])
    ocp.model.con_h_expr_0 = ocp.model.con_h_expr

    nh = 2
    ocp.constraints.idxsh = np.arange(2)
    ocp.constraints.lh = np.zeros((nh, ))
    ocp.constraints.uh = np.zeros((nh, ))
    ocp.cost.zl = 1e0 * np.ones((nh, ))
    ocp.cost.zu = 1e0 * np.ones((nh, ))
    ocp.cost.Zl = 0e0 * np.ones((nh, ))
    ocp.cost.Zu = 0e0 * np.ones((nh, ))

    ocp.constraints.idxsh_0 = np.arange(2)
    ocp.constraints.lh_0 = np.zeros((nh, ))
    ocp.constraints.uh_0 = np.zeros((nh, ))
    ocp.cost.zl_0 = 1e0 * np.ones((nh, ))
    ocp.cost.zu_0 = 1e0 * np.ones((nh, ))
    ocp.cost.Zl_0 = 0e0 * np.ones((nh, ))
    ocp.cost.Zu_0 = 0e0 * np.ones((nh, ))

    # # add obstacle
    # circular_constraints = unicycle_get_circular_constraints()
    # for x, y, r in circular_constraints:
    #     dist = ca.sqrt((model.x[0] - x) ** 2 + (model.x[1] - y) ** 2 + 1e-3)
    #     ocp.formulate_constraint_as_L2_penalty(dist, 1e6, lower_bound=r, upper_bound=None)

    X0 = np.array([1.0, 1.0, 0.0, np.pi, 0.0, 0.0, 0.0])  # Intital state
    ocp.constraints.x0 = X0[:nx]

    # set options
    set_solver_options_in_ocp(ocp, options)

    return ocp

