from pymoo.factory import get_performance_indicator
#from pymoo.indicators.hv import HV
import numpy as np

def hypervolumn(A, ref=None, type='acc'):
    """
    :param A: np.array, num_points, num_task
    :param ref: num_task
    """
    dim = A.shape[1]

    if type == 'acc':
        if ref is None:
            ref = np.zeros(dim)
        hv = HV(ref_point=ref)
        return hv.do(-A)

    elif type == 'loss':
        if ref is None:
            ref = np.ones(dim)
        hv = get_performance_indicator("hv",ref_point=ref)
        return hv.do(A)
    else:
        print('type not implemented')
        return None

def get_pareto_front(A, type='acc'):
    """
    Get dominated point
    :param A: np.array, num_points, dim
    :param ref: dim
    """
    if A.shape[0] == 0:
        return A

    pareto_front = []
    if type == 'acc':
        for _ in range(A.shape[0]):
            if not (((A - A[_, :]) > 0).sum(axis=1) == A.shape[1]).sum():
                pareto_front.append(list(A[_, :]))
    elif type == 'loss':
        for _ in range(A.shape[0]):
            if not (((A - A[_, :]) < 0).sum(axis=1) == A.shape[1]).sum():
                pareto_front.append(list(A[_, :]))

    return pareto_front
    # return np.array(pareto_front)