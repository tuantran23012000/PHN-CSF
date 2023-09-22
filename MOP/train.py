import sys
import os
sys.path.append(os.getcwd())
import time
from tqdm import tqdm
import numpy as np
import torch
from tools.scalarization_function import CS_functions,EPOSolver
from tools.hv import HvMaximization
from models.hyper_trans import Hyper_trans
from models.hyper_trans_posi import Hyper_trans_posi
from models.hyper_mlp import Hyper_mlp
from eval import eval
from test import test
from tools.metrics import IGD, MED
import copy
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
def train_epoch(device, cfg, criterion, pb,pf,model_type):
    print(model_type)
    name = cfg['NAME']
    mode = cfg['MODE']
    if model_type == 'mlp':
        # ray_hidden_dim = cfg['TRAIN']['Ray_hidden_dim_mlp']
        ray_hidden_dims = [2*(i+1) for i in range(4,150)]
    else:
        # ray_hidden_dim = cfg['TRAIN']['Ray_hidden_dim_trans']
        ray_hidden_dims = [2*(i+1) for i in range(4,50)]
    out_dim = cfg['TRAIN']['Out_dim']
    n_tasks = cfg['TRAIN']['N_task']
    num_hidden_layer = cfg['TRAIN']['Solver'][criterion]['Num_hidden_layer']
    last_activation = cfg['TRAIN']['Solver'][criterion]['Last_activation']
    ref_point = tuple(map(int, cfg['TRAIN']['Ref_point'].split(',')))
    lr = cfg['TRAIN']['OPTIMIZER']['Lr']
    wd = cfg['TRAIN']['OPTIMIZER']['WEIGHT_DECAY']
    type_opt = cfg['TRAIN']['OPTIMIZER']['TYPE']
    epochs = cfg['TRAIN']['Epoch']
    iters = cfg['TRAIN']['Iter']
    alpha_r = cfg['TRAIN']['Alpha']
    start = 0.
    if n_tasks == 2:
        rays_train = np.load("./datasets/train/train_rays_2d.npy")
        rays_eval = np.load("./datasets/val/eval_rays_2d.npy")
        rays_test = np.load("./datasets/test/test_rays_2d.npy")
        
    else:
        rays_train = np.load("./datasets/train/train_rays_3d.npy")
        rays_eval = np.load("./datasets/val/eval_rays_3d.npy")
        rays_test = np.load("./datasets/test/test_rays_3d.npy")
    print("Train size: ",rays_train.shape)
    print("Val size: ",rays_eval.shape)
    print("Test size: ",rays_test.shape)
    #rays_train = torch.from_numpy(rays_train).float()
    #rays_eval = torch.from_numpy(rays_eval).float()
    #train_dt = torch.utils.data.TensorDataset(rays_train)
    #val_dt = torch.utils.data.TensorDataset(rays_eval)
    
    # train_loader = torch.utils.data.DataLoader(
    #     dataset=train_dt,
    #     batch_size=1,num_workers=4,
    #     shuffle=True)
    # val_loader = torch.utils.data.DataLoader(
    #     dataset=train_dt,
    #     batch_size=1,num_workers=4,
    #     shuffle=False)
    
    PARAMS = []
    MEDS = []
    for ray_hidden_dim in tqdm(ray_hidden_dims):
        if model_type == 'mlp':
            hnet = Hyper_mlp(ray_hidden_dim = ray_hidden_dim, out_dim = out_dim, n_tasks = n_tasks,num_hidden_layer=num_hidden_layer,last_activation=last_activation)
            hnet_copy = Hyper_mlp(ray_hidden_dim = ray_hidden_dim, out_dim = out_dim, n_tasks = n_tasks,num_hidden_layer=num_hidden_layer,last_activation=last_activation)
        elif model_type == 'trans':
            hnet = Hyper_trans(ray_hidden_dim = ray_hidden_dim, out_dim = out_dim, n_tasks = n_tasks,num_hidden_layer=num_hidden_layer,last_activation=last_activation)
            hnet_copy = Hyper_trans(ray_hidden_dim = ray_hidden_dim, out_dim = out_dim, n_tasks = n_tasks,num_hidden_layer=num_hidden_layer,last_activation=last_activation)
        else:
            hnet = Hyper_trans_posi(ray_hidden_dim = ray_hidden_dim, out_dim = out_dim, n_tasks = n_tasks,num_hidden_layer=num_hidden_layer,last_activation=last_activation)
            hnet_copy = Hyper_trans_posi(ray_hidden_dim = ray_hidden_dim, out_dim = out_dim, n_tasks = n_tasks,num_hidden_layer=num_hidden_layer,last_activation=last_activation)
        hnet = hnet.to(device)
        hnet_copy = hnet_copy.to(device)
        param = count_parameters(hnet)
        PARAMS.append(param)
        #print("Model size: ",count_parameters(hnet))
        sol = []
        if type_opt == 'adam':
            optimizer = torch.optim.Adam(hnet.parameters(), lr = lr, weight_decay=wd) 
        elif type_opt == 'adamw':
            optimizer = torch.optim.AdamW(hnet.parameters(), lr = lr, weight_decay=wd)
        mo_opt = HvMaximization(1, n_tasks, ref_point)

        print("Dim: ",ray_hidden_dim)
        best_med = 1000
        for epoch in range(epochs):
            for r in rays_train:        
                ray = torch.Tensor(r.tolist()).to(device)
                hnet.train()
                optimizer.zero_grad()
                output = hnet(ray)
                if name == 'ex3':
                    output = torch.sqrt(output)
                if epoch == epochs - 1:
                    sol.append(output.cpu().detach().numpy().tolist()[0])
                ray_cs = 1/ray
                ep_ray = 1.1 * ray_cs / np.linalg.norm(ray_cs)
                ray = ray.squeeze(0)
                obj_values = []
                objectives = pb.get_values(output)
                for i in range(len(objectives)):
                    obj_values.append(objectives[i][0])
                losses = torch.stack(obj_values)
                CS_func = CS_functions(losses,ray)

                if criterion == 'Prod':
                    loss = CS_func.product_function()
                elif criterion == 'Log':
                    loss = CS_func.log_function()
                elif criterion == 'AC':
                    rho = cfg['TRAIN']['Solver'][criterion]['Rho']
                    loss = CS_func.ac_function(rho = rho)
                elif criterion == 'MC':
                    rho = cfg['TRAIN']['Solver'][criterion]['Rho']
                    loss = CS_func.mc_function(rho = rho)
                elif criterion == 'HV':
                    loss_numpy = []
                    for j in range(1):
                        loss_numpy.append(losses.detach().cpu().numpy())
                    loss_numpy = np.array(loss_numpy).T
                    loss_numpy = loss_numpy[np.newaxis, :, :]
                    rho = cfg['TRAIN']['Solver'][criterion]['Rho']
                    dynamic_weight = mo_opt.compute_weights(loss_numpy[0,:,:])
                    loss = CS_func.hv_function(dynamic_weight.reshape(1,3),rho = rho)
                elif criterion == 'LS':
                    loss = CS_func.linear_function()
                elif criterion == 'Cheby':
                    loss = CS_func.chebyshev_function()
                elif criterion == 'Utility':
                    ub = cfg['TRAIN']['Solver'][criterion]['Ub']
                    loss = CS_func.utility_function(ub = ub)
                elif criterion == 'KL':
                    loss = CS_func.KL_function()
                elif criterion == 'Cosine':
                    loss = CS_func.cosine_function()
                elif criterion == 'Cauchy':
                    CS_func = CS_functions(losses,ray_cs)
                    loss = CS_func.cauchy_schwarz_function()
                elif criterion == 'EPO':
                    solver = EPOSolver(n_tasks=n_tasks, n_params=count_parameters(hnet))
                    loss = solver(losses, ray, list(hnet.parameters()))
                loss.backward()
                optimizer.step()
            results, targets = eval(hnet,criterion,pb,pf,cfg,rays_eval,device)
            targets = np.array(targets, dtype='float32')
            results = np.array(results, dtype='float32')
            med = MED(targets, results)
            if med < best_med:
                best_med = med
                # print("Epoch:", epoch)
                # print("MED: ",best_med)
                hnet_copy = copy.deepcopy(hnet)
                #print(hnet_copy)
                # if model_type == 'mlp':
                #     torch.save(hnet,("./save_weights/best_weight_"+str(criterion)+"_"+str(mode)+"_"+str(name)+"_" + str(ray_hidden_dim)+".pt"))
                # elif model_type == 'trans':
                #     torch.save(hnet,("./save_weights/best_weight_"+str(criterion)+"_"+str(mode)+"_"+str(name)+"_" + str(ray_hidden_dim)+"_at.pt"))
                # else:
                #     torch.save(hnet,("./save_weights/best_weight_"+str(criterion)+"_"+str(mode)+"_"+str(name)+"_" + str(ray_hidden_dim)+"_at_position.pt"))
        results_test, targets_test = test(hnet_copy,criterion,pb,pf,cfg,rays_eval,device)   
        med_test = MED(targets_test, results_test)
        print("PARAM:{}, MED:{}".format(str(param),str(med_test)))
        MEDS.append(med_test)
    MEDS = np.array(MEDS)
    PARAMS = np.array(PARAMS)
    return MEDS,PARAMS
