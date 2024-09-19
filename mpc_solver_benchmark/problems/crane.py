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


from acados_template import AcadosOcp, AcadosModel, AcadosSimSolver
import numpy as np
from casadi import SX, vertcat, diag
from dataclasses import dataclass


@dataclass
class CraneOcpOptions:
    qp_solver: str = "PARTIAL_CONDENSING_HPIPM"
    sim_method_num_steps: int = 3
    sim_method_num_stages: int = 4
    N_horizon: int = 7
    with_anderson_acceleration: bool = False
    x0 = np.array([2.0, 0.0, 2.0, 0.0])
    xf = np.array([0., 0., 0., 0.])


def formulate_crane_ocp(params: CraneOcpOptions) -> AcadosOcp:

    # (very) simple crane model
    beta = 0.001
    k = 0.9
    a_max = 10
    dt_max = 2.0
    dt_min = 1e-3

    # states
    p1 = SX.sym('p1')
    v1 = SX.sym('v1')
    p2 = SX.sym('p2')
    v2 = SX.sym('v2')

    x = vertcat(p1, v1, p2, v2)

    # controls
    a = SX.sym('a')
    dt = SX.sym('dt')

    u = vertcat(a, dt)

    f_expl = dt*vertcat(v1, a, v2, -beta*v2-k*(p2 - p1))

    model = AcadosModel()

    model.f_expl_expr = f_expl
    model.x = x
    model.u = u
    model.name = 'crane_time_opt'

    ocp = AcadosOcp()
    ocp.model = model
    ocp.solver_options.N_horizon = params.N_horizon
    ocp.solver_options.tf = params.N_horizon

    ocp.cost.cost_type = 'EXTERNAL'
    ocp.cost.cost_type_e = 'EXTERNAL'

    ocp.model.cost_expr_ext_cost = dt
    ocp.model.cost_expr_ext_cost_e = 0

    ocp.model.cost_expr_ext_cost_custom_hess = diag(vertcat(SX.zeros(1, 1), 1./(dt), SX.zeros(model.x.rows(), 1)))

    ocp.constraints.lbu = np.array([-a_max, dt_min])
    ocp.constraints.ubu = np.array([+a_max, dt_max])
    ocp.constraints.idxbu = np.array([0, 1])

    ocp.constraints.x0 = params.x0
    ocp.constraints.lbx_e = params.xf
    ocp.constraints.ubx_e = params.xf
    ocp.constraints.idxbx_e = np.array([0, 1, 2, 3])

    # set options
    ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'#'PARTIAL_CONDENSING_HPIPM' # FULL_CONDENSING_QPOASES
    ocp.solver_options.integrator_type = 'ERK'
    ocp.solver_options.nlp_solver_type = 'SQP'
    ocp.solver_options.globalization = 'MERIT_BACKTRACKING'
    ocp.solver_options.alpha_min = 0.01
    ocp.solver_options.nlp_solver_max_iter = 50
    ocp.solver_options.nlp_solver_tol_stat = 1e-7
    ocp.solver_options.qp_solver_tol_stat = 1e-1*ocp.solver_options.nlp_solver_tol_stat
    ocp.solver_options.sim_method_num_steps = 15
    ocp.solver_options.qp_solver_iter_max = 50
    ocp.solver_options.hessian_approx = 'EXACT'
    ocp.solver_options.exact_hess_constr = 0
    ocp.solver_options.exact_hess_dyn = 0
    ocp.solver_options.log_primal_step_norm = True
    ocp.solver_options.with_anderson_acceleration = params.with_anderson_acceleration

    return ocp

def simulate_on_fine_grid(integrator: AcadosSimSolver, simU: np.ndarray, simX: np.ndarray):

    dts = simU[:, 1]
    N = simU.shape[0]
    nu = simU.shape[1] - 1
    nx = simX.shape[1]

    # simulate on finer grid
    dt_approx = 0.0005

    dts_fine = np.zeros((N,))
    Ns_fine = np.zeros((N,), dtype='int16')

    # compute number of simulation steps for bang interval + dt_fine
    for i in range(N):
        N_approx = max(int(dts[i]/dt_approx), 1)
        dts_fine[i] = dts[i]/N_approx
        Ns_fine[i] = int(round(dts[i]/dts_fine[i]))

    N_fine = int(np.sum(Ns_fine))

    simU_fine = np.zeros((N_fine, nu))
    ts_fine = np.zeros((N_fine+1, ))
    simX_fine = np.zeros((N_fine+1, nx))
    simX_fine[0, :] = simX[0, :]

    k = 0
    for i in range(N):
        u = simU[i, 0]
        integrator.set("u", np.hstack((u, np.ones(1, ))))
        integrator.set("T", dts_fine[i])

        for _ in range(Ns_fine[i]):
            integrator.set("x", simX_fine[k,:])
            status = integrator.solve()
            if status != 0:
                raise Exception(f'acados returned status {status}.')

            simX_fine[k+1,:] = integrator.get("x")
            simU_fine[k, :] = u
            ts_fine[k+1] = ts_fine[k] + dts_fine[i]

            k += 1

    return ts_fine, simU_fine, simX_fine

