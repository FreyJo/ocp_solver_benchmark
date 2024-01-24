from dataclasses import dataclass
from typing import List, Union, Optional
import pickle
import numpy as np


@dataclass
class SimulationResultsBase:
    x_traj: Union[np.ndarray, List[np.ndarray]]
    u_traj: Union[np.ndarray, List[np.ndarray]]
    t_traj: np.ndarray
    y_ref_traj: Optional[np.ndarray] = None
    time_stamp: Optional[str] = None
    time_tot: Optional[np.ndarray] = None
    time_sim: Optional[np.ndarray] = None
    time_lin: Optional[np.ndarray] = None
    time_qp_solver_call: Optional[np.ndarray] = None
    time_qp: Optional[np.ndarray] = None
    time_reg: Optional[np.ndarray] = None
    time_preparation: Optional[np.ndarray] = None
    time_feedback: Optional[np.ndarray] = None
    qp_iter: Optional[np.ndarray] = None
    status: Optional[np.ndarray] = None

    def save_to_file(self, filename: str):
        print(f'Saving results to {filename}.')
        with open(filename, 'wb') as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)


@dataclass
class SimulationResultsOpenLoop(SimulationResultsBase):
    y_ref_terminal: Optional[np.ndarray] = None
    nlp_iter: Optional[int] = None
    time_py: Optional[np.ndarray] = None
    primal_step_norm_traj: Optional[np.ndarray] = None
    cost_value: Optional[float] = None


@dataclass
class SimulationResultsClosedLoop(SimulationResultsBase):
    cost_traj: Optional[np.ndarray] = None
    nlp_iter: Optional[np.ndarray] = None


def load_from_file(filename: str):
    # print(f'Loading results from {filename}.')
    with open(filename, 'rb') as handle:
        obj = pickle.load(handle)
    return obj

def get_results_from_filenames(filenames: List[str]) -> List[SimulationResultsBase]:
    results = []
    for filename in filenames:
        results.append(load_from_file(filename))
    return results
