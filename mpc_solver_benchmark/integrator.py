from abc import abstractmethod, ABC
import numpy as np
import casadi as ca

from acados_template import AcadosOcp, AcadosSimSolver, AcadosSim, create_model_with_cost_state

class BaseIntegrator(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def simulate(self, x0: np.ndarray, u: np.ndarray, t: float):
        pass


class AcadosIntegrator(BaseIntegrator):

    def __init__(self, ref_ocp: AcadosOcp):
        super().__init__()
        sim = AcadosSim()
        model, parameter_values = create_model_with_cost_state(ref_ocp)
        sim.model = model
        sim.parameter_values = parameter_values

        self.has_t0_param = model.t0 is not None
        if self.has_t0_param:
            wd = ca.which_depends(model.p, model.t0)
            self.t0_param_idx = next((i for i, w in enumerate(wd) if w))

        # copy discretization options from OCP
        if ref_ocp.solver_options.time_steps is not None:
            sim.solver_options.T = ref_ocp.solver_options.time_steps[0]
        else:
            sim.solver_options.T = ref_ocp.solver_options.tf / ref_ocp.dims.N
        sim.solver_options.integrator_type = ref_ocp.solver_options.integrator_type
        sim.solver_options.collocation_type = ref_ocp.solver_options.collocation_type
        sim.solver_options.num_stages = ref_ocp.solver_options.sim_method_num_stages
        sim.solver_options.num_steps = ref_ocp.solver_options.sim_method_num_steps
        sim.solver_options.sim_method_jac_reuse = ref_ocp.solver_options.sim_method_jac_reuse
        sim.solver_options.newton_iter = ref_ocp.solver_options.sim_method_newton_iter
        # integrator specific options
        sim.solver_options.sens_forw = False
        sim.solver_options.sens_adj = False
        sim.solver_options.sens_algebraic = False
        sim.solver_options.sens_hess = False
        sim.solver_options.output_z = False
        # create solver
        self.solver = AcadosSimSolver(sim)

    def simulate(self, x0: np.ndarray, u: np.ndarray, t: float):
        if self.has_t0_param:
            p_val = self.solver.acados_sim.parameter_values.copy()
            p_val[self.t0_param_idx] = t
        else:
            p_val = None
        return self.solver.simulate(x0, u=u, p=p_val)

    @property
    def T(self,) -> float:
        return self.solver.T


