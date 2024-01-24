
import numpy as np
import casadi as ca
from copy import deepcopy
from typing import Union
from acados_template import AcadosOcp
from scipy.linalg import solve_continuous_are


def reformulate_as_clc_ocp(ocp: AcadosOcp, K: np.ndarray, x_ref: Union[np.ndarray, ca.SX], u_ref: Union[np.ndarray, ca.SX]):
    """Reformulate ocp to closed-loop costing ocp by eliminating the controls via a fixed linear control law K"""

    if not ocp.cost.cost_type == ocp.cost.cost_type_e == "NONLINEAR_LS":
        raise Exception("Not implemented.")

    # set up feedback law
    K_ = ca.sparsify(ca.DM(K))

    if isinstance(x_ref, np.ndarray):
        x_ref = ca.sparsify(ca.DM(x_ref))
    if isinstance(u_ref, np.ndarray):
        u_ref = ca.sparsify(ca.DM(u_ref))
    # feedback_law = u_ref + K_ @ (ocp.model.x - x_ref)
    feedback_law = u_ref

    # substitute u with feedback law in model
    ocp.model.substitute(ocp.model.u, feedback_law)

    # remove u
    ocp.model.u = ca.SX.sym('u', 0)
    # remove u constraints
    ocp.constraints.lbu = np.array([])
    ocp.constraints.ubu = np.array([])
    ocp.constraints.idxbu = np.array([])

    return ocp


def compute_lqr_gain_continuous_time(model, model_params, mpc_params):

    # linearize dynamics
    A_sym_fun = ca.Function(
        "A_sym_fun", [model.x, model.u], [ca.jacobian(model.f_expl_expr, model.x)]
    )
    B_sym_fun = ca.Function(
        "B_sym_fun", [model.x, model.u], [ca.jacobian(model.f_expl_expr, model.u)]
    )

    A_mat = A_sym_fun(model_params.xs, model_params.us).full()
    B_mat = B_sym_fun(model_params.xs, model_params.us).full()

    P_mat = solve_continuous_are(A_mat, B_mat, mpc_params.Q, mpc_params.R)
    K_mat = np.linalg.inv(mpc_params.R) @ B_mat.T @ P_mat
    print(f"P_mat {P_mat}")
    return P_mat, K_mat