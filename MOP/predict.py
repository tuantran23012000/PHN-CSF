import torch
import os
import sys
sys.path.append(os.getcwd())
import numpy as np
import random
import argparse
from tools.utils import find_target, circle_points_random, get_d_paretomtl
from tools.utils import circle_points, sample_vec
from tools.metrics import IGD, MED

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
def get_rays(cfg,num_ray_init):
    contexts = np.array(sample_vec(cfg['TRAIN']['N_task'],num_ray_init))
    tmp = []
    for r in contexts:
        flag = True
        for i in r:
            if i <=0.16:
                flag = False
                break
        if flag:

            tmp.append(r)
    contexts = np.array(tmp)
    return contexts
def predict_result(device,cfg,criterion,pb,pf,num_e=None,contexts = []):
    mode = cfg['MODE']
    name = cfg['NAME']
    num_ray_init = cfg['EVAL']['Num_ray_init']
    num_ray_test = cfg['EVAL']['Num_ray_test']
    hnet1 = torch.load("./save_weights/best_weight_"+str(criterion)+"_"+str(mode)+"_"+str(name)+"_"+ str(cfg['TRAIN']['Ray_hidden_dim'])+".pt",map_location=device)
    hnet1.eval()
    results1 = []
    targets_epo = []
    contexts = get_rays(cfg, num_ray_init)
    rng = np.random.default_rng()
    contexts = rng.choice(contexts,num_ray_test)
    #contexts = np.array([[0.2, 0.5,0.3], [0.4, 0.25,0.35],[0.3,0.2,0.5],[0.55,0.2,0.25]])
    for r in contexts:
        r_inv = 1. / r
        ray = torch.Tensor(r.tolist()).to(device)
        output = hnet1(ray)
        output = torch.sqrt(output)
        objectives = pb.get_values(output)
        obj_values = []
        for i in range(len(objectives)):
            obj_values.append(objectives[i].cpu().detach().numpy().tolist())
        results1.append(obj_values)
        if criterion == "Cauchy":
            target_epo = find_target(pf, criterion = criterion, context = r_inv.tolist(),cfg=cfg)
        else:
            target_epo = find_target(pf, criterion = criterion, context = r.tolist(),cfg=cfg)
        targets_epo.append(target_epo)

    targets_epo = np.array(targets_epo)
    tmp = []

    results1 = np.array(results1, dtype='float32')
    med = np.mean(np.sqrt(np.sum(np.square(targets_epo-results1),axis = 1)))
    print("MED: ",med)
    med = MED(targets_epo, results1)
    
    d_i = []
    for target in pf:
        d_i.append(np.min(np.sqrt(np.sum(np.square(target-results1),axis = 1))))
    igd = np.mean(np.array(d_i))
    
    igd = IGD(pf, results1)
    print("IGD:",igd)
    return igd, targets_epo, results1, contexts, med