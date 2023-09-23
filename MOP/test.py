import torch
import numpy as np
from tools.utils import find_target
def test(hnet,criterion,pb,pf,cfg,test_loader,device):
    hnet.eval()
    name = cfg['NAME']
    results = []
    targets = []
    for i, batch in enumerate(test_loader):
        ray =  batch[0].squeeze(0).to(device)
        r = ray.detach().cpu().numpy()
        output = hnet(ray)
        if name == 'ex3':
            output = torch.sqrt(output)
        objectives = pb.get_values(output)
        obj_values = []
        for i in range(len(objectives)):
            #print(objectives[i].cpu().detach().numpy().tolist())
            obj_values.append(objectives[i].cpu().detach().numpy().tolist())
        # print(obj_values)
        obj_values = np.array(obj_values).T
        results.append(obj_values)

        target = find_target(pf, criterion = criterion, context = r.tolist(),cfg=cfg)
        target = np.array(target)
        targets.append(target)
    return results, targets