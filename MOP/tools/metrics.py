import numpy as np
import torch

def MED(target_points, predict_points):
    med = np.mean(np.sqrt(np.sum(np.square(target_points-predict_points),axis = 1)))
    return med
    
def IGD(pf_truth, pf_approx):
    d_i = []
    for pf in pf_truth:
        d_i.append(np.min(np.sqrt(np.sum(np.square(pf-pf_approx),axis = 1))))
    igd = np.mean(np.array(d_i))
    return igd