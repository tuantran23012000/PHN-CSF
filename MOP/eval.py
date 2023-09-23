import torch
import numpy as np
from tools.utils import find_target
def eval(hnet,criterion,pb,pf,cfg,val_loader,device):
    hnet.eval()
    results = []
    targets = []
    name = cfg['NAME']
    for i, batch in enumerate(val_loader):
        ray =  batch[0].squeeze(0).to(device)
        r = ray.detach().cpu().numpy()
        output = hnet(ray)
        if name == 'ex3':
            output = torch.sqrt(output)
        objectives = pb.get_values(output)
        obj_values = []
        for i in range(len(objectives)):
            obj_values.append(objectives[i].cpu().detach().numpy().tolist())
        obj_values = np.array(obj_values).T
        results.append(obj_values)

        target = find_target(pf, criterion = criterion, context = r,cfg=cfg)
        target = np.array(target)
        targets.append(target)
    return results, targets