from typing import Union
import numpy as np
import casadi as ca
from .solver_base import BaseSolver
from .reference import Reference
from acados_template import AcadosOcp, AcadosMultiphaseOcp, AcadosOcpSolver

class AcadosSolver(BaseSolver):

    def __init__(self, ocp: Union[AcadosOcp, AcadosMultiphaseOcp], filename: str = 'solver.json', with_x0: bool = True):
        super().__init__()
        self.ocp = ocp
        self.solver = AcadosOcpSolver(ocp, json_file = filename, verbose = False)

        if ocp.solver_options.nlp_solver_type == 'SQP_RTI':
            self.is_real_time_solver = True

        if with_x0:
            if isinstance(ocp, AcadosOcp) and ocp.constraints.has_x0:
                self.__x0 = ocp.constraints.x0
            elif isinstance(ocp, AcadosMultiphaseOcp) and ocp.constraints[0].has_x0:
                self.__x0 = ocp.constraints[0].x0
            else:
                raise Exception("Initial state x0 is not defined in the constraints.")

    def _prepare_solve(self, x0: np.ndarray, t: float):
        pass

    def solve_for_x0(self, x0: np.ndarray, t: float) -> np.ndarray:
        self.__x0 = x0
        self._prepare_solve(x0, t)
        u0 = self.solver.solve_for_x0(x0, fail_on_nonzero_status=False)
        return u0

    def solve(self):
        self.solver.solve()

    def real_time_preparation(self, t: float):
        self._prepare_solve(self.__x0, t)
        self.solver.options_set("rti_phase", 1)
        self.solver.solve()

    def real_time_feedback(self, x0: np.ndarray, t: float) -> np.ndarray:
        self.__x0 = x0
        self.solver.set(0, "lbx", x0)
        self.solver.set(0, "ubx", x0)
        self.solver.options_set("rti_phase", 2)
        self.solver.solve()
        return self.solver.get(0, "u")

    def get_stats(self, field: str):
        return self.solver.get_stats(field)

    def reset(self):
        self.solver.reset()

    def print_statistics(self, ):
        self.solver.print_statistics()

    def get_nx(self,):
        return self.solver.get(0, 'x').shape[0]

    def get_nu(self,):
        return self.solver.get(0, 'u').shape[0]

    def get_N_horizon(self,):
        return self.solver.N

    def get_t_traj(self, ):
        return self.solver.acados_ocp.solver_options.shooting_nodes

    def get_x_traj(self,):
        x_traj = []
        for i in range(self.solver.N+1):
            x_traj.append(self.solver.get(i, "x"))
        return x_traj

    def set_x_init_traj(self, x_traj):
        for i, x_ in enumerate(x_traj):
            self.solver.set(i, "x", x_)

    def get_u_traj(self,):
        u_traj = []
        for i in range(self.solver.N):
            u_traj.append(self.solver.get(i, "u"))
        return u_traj

    def set_u_init_traj(self, u_traj):
        for i, u_ in enumerate(u_traj):
            self.solver.set(i, "u", u_)

    def get_y_reference(self, t):
        pass

    def store_iterate(self, filename: str = '', overwrite: bool = True):
        self.solver.store_iterate(filename, overwrite, verbose=False)

    def load_iterate(self, filename: str,):
        self.solver.load_iterate(filename, verbose=False)

    def get_status(self):
        return self.solver.get_status()

    def get_cost(self):
        return self.solver.get_cost()


class AcadosSolverConstantReference(AcadosSolver):
    """
    An AcadosSolver which has a constant reference over the horizon at each time step.
    I.e. this controller is not aware of future jumps in the reference trajectory.
    """
    def __init__(self, ocp: AcadosOcp, reference: Reference, reference_terminal: Reference, filename: str = 'solver.json'):
        super().__init__(ocp, filename)

        self.reference = reference
        self.reference_terminal = reference_terminal

        self.has_reference = True


    def get_y_reference_traj(self,):
        y_ref_traj = np.zeros((self.solver.N, self.solver.reference.ny))

        ts = self.get_t_traj()
        for i in range(self.solver.N):
            y_ref_traj[i, :] = self.reference.get_reference(ts[i])

        y_ref_terminal = self.reference_terminal.get_reference(ts[-1])
        return y_ref_traj, y_ref_terminal


    def _prepare_solve(self, x0: np.ndarray, t: float):
        y_ref = self.reference.get_reference(t)
        y_ref_e = self.reference_terminal.get_reference(t)

        for i in range(self.solver.N):
            self.solver.set(i, 'yref', y_ref)

        self.solver.set(self.solver.N, 'yref', y_ref_e)



class AcadosSolverTimeVaryingReference(AcadosSolverConstantReference):

    def _prepare_solve(self, x0: np.ndarray, t: float):

        for i in range(self.solver.N):
            y_ref = self.reference.get_reference(t)
            self.solver.set(i, 'yref', y_ref)
            t += self.solver.acados_ocp.solver_options.time_steps[i]

        y_ref_e = self.reference_terminal.get_reference(t)
        self.solver.set(self.solver.N, 'yref', y_ref_e)


class AcadosSolverTimeVarying(AcadosSolver):

    '''An AcadosSolver which has a parameter t0 that can be used to set time-varying references.'''

    def __init__(self, ocp: AcadosOcp, filename: str = 'solver.json'):
        super().__init__(ocp, filename)

        if isinstance(ocp, AcadosOcp):
            wd = ca.which_depends(ocp.model.p, ocp.model.t0)
        else:
            wd = ca.which_depends(ocp.model[0].p, ocp.model[0].t0)

        self.t0_param_idx = next((i for i, w in enumerate(wd) if w))

    def _prepare_solve(self, x0: np.ndarray, t: float):
        for i in range(self.solver.N):
            self.solver.set_params_sparse(i, np.array([self.t0_param_idx]), np.array([t]))
            t += self.solver.acados_ocp.solver_options.time_steps[i]
        self.solver.set_params_sparse(self.solver.N, np.array([0]), np.array([t]))

