import torch
import numpy as np
from tools.utils import find_target
def test(hnet,criterion,pb,pf,cfg,rays_test,device):
    hnet.eval()
    name = cfg['NAME']
    results = []
    targets = []
    for r in rays_test:
        r_inv = 1. / r
        ray = torch.Tensor(r.tolist()).to(device)
        output = hnet(ray)
        if name == 'ex3':
            output = torch.sqrt(output)
        objectives = pb.get_values(output)
        obj_values = []
        for i in range(len(objectives)):
            #print(objectives[i].cpu().detach().numpy().tolist())
            obj_values.append(objectives[i].cpu().detach().numpy().tolist()[0])
        # print(obj_values)
        results.append(obj_values)
        if criterion == "Cauchy":
            target= find_target(pf, criterion = criterion, context = r_inv.tolist(),cfg=cfg)
        else:
            target = find_target(pf, criterion = criterion, context = r.tolist(),cfg=cfg)
        targets.append(target)

    targets= np.array(targets, dtype='float32')
    results = np.array(results, dtype='float32')
    # med = MED(targets, results)
    # meds.append(med)
    # print("MED: ",med)
    # d_i = []
    # for target in pf:
    #     d_i.append(np.min(np.sqrt(np.sum(np.square(target-results1),axis = 1))))
    # igd = np.mean(np.array(d_i))
    
    # igd = IGD(pf, results1)
    # print("IGD:",igd)
    # MEDS = np.array(meds)
    # PARAMS = np.array(params)
    # np.save("med_"+str(model_type)+".npy",MEDS)
    # np.save("param_"+str(model_type)+".npy",PARAMS)
    return results, targets