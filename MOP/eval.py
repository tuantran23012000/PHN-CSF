import torch
import numpy as np
from tools.utils import find_target
def eval(hnet,criterion,pb,pf,cfg,rays_eval,device):
    hnet.eval()
    results = []
    targets = []
    name = cfg['NAME']
    for r in rays_eval:
    # for i, batch in enumerate(train_loader):
    #     ray =  batch[0].squeeze(0)
    #     ray.detach().cpu().numpy().tolist()
        ray = torch.Tensor(r.tolist()).to(device)
        output = hnet(ray)
        if name == 'ex3':
            output = torch.sqrt(output)
        objectives = pb.get_values(output)
        obj_values = []
        for i in range(len(objectives)):
            obj_values.append(objectives[i].cpu().detach().numpy().tolist()[0])
        results.append(obj_values)

        target = find_target(pf, criterion = criterion, context = r.tolist(),cfg=cfg)
        targets.append(target)
    return results, targets