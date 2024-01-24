from typing import List, Union
import numpy as np
import casadi as ca

class Reference():

    def __init__(self, t_jump: List[float] = [], references: List[np.ndarray] = [np.zeros((0,))]):
        """
        parameters:
        t_jump: sorted list of time points where the reference changes
        references: list of references values
        """
        self.t_jump = t_jump
        self.references = references
        self.ny = references[0].shape[0]

    def get_reference(self, t: float):
        for i, jump in enumerate(self.t_jump):
            if jump > t:
                return self.references[i]
        return self.references[-1]

    def get_casadi_expression(self, t_expr: Union[ca.SX, ca.MX, float]):

        y_ref_expr = self.references[0]

        if len(self.t_jump) > 0:
            for t, ref_next in zip(self.t_jump[:], self.references[1:]):
                y_ref_expr = ca.if_else(t_expr < t, y_ref_expr, ref_next)
        return y_ref_expr

    def get_sub_reference(self, idxy: List[int]):
        sub_ref_list = [r[idxy] for r in self.references]
        return Reference(t_jump=self.t_jump, references=sub_ref_list)


if __name__ == '__main__':

    yref = [2, 4, 6]
    t_jump = [1, 2]

    ref = Reference(t_jump, yref)

    t_test = 2.5
    print(ref.get_casadi_expression(t_test))
    print(ref.get_reference(t_test))

    t_test = 1.
    print(ref.get_casadi_expression(t_test))
    print(ref.get_reference(t_test))
