from abc import abstractmethod, ABC
import numpy as np

class BaseSolver(ABC):

    def __init__(self):
        self.has_reference = False
        self.is_real_time_solver = False

    @abstractmethod
    def solve_for_x0(self, x0: np.ndarray, t: float):
        pass


    @abstractmethod
    def solve(self):
        pass

    @abstractmethod
    def real_time_preparation(self, t: float):
        pass

    @abstractmethod
    def real_time_feedback(self, t: float):
        pass

    @abstractmethod
    def get_stats(self, field: str):
        pass

    @abstractmethod
    def reset(self, ):
        pass

    @abstractmethod
    def print_statistics(self, ):
        pass

    @abstractmethod
    def get_nx(self,):
        pass

    @abstractmethod
    def get_nu(self,):
        pass

    @abstractmethod
    def get_N_horizon(self,):
        pass

    @abstractmethod
    def get_t_traj(self, ):
        pass

    @abstractmethod
    def get_x_traj(self,):
        pass

    @abstractmethod
    def get_u_traj(self,):
        pass

    @abstractmethod
    def set_x_init_traj(self,):
        pass

    @abstractmethod
    def set_u_init_traj(self,):
        pass

    @abstractmethod
    def store_iterate(self, filename: str = '', overwrite: bool = True):
        pass

    @abstractmethod
    def load_iterate(self, filename: str,):
        pass

    @abstractmethod
    def get_status(self,) -> int:
        pass

    @abstractmethod
    def get_stats(self, field: str):
        pass

    @abstractmethod
    def get_cost(self):
        pass

    def get_y_reference_traj(self,):
        return (None, None)

    def get_y_reference(self, t):
        return None