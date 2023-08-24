import sys
import os
sys.path.append(os.getcwd())
import time
from tqdm import tqdm
from model import Hypernetwork
import numpy as np
import torch
from tools.scalarization_function import CS_functions,EPOSolver
from tools.hv import HvMaximization
def train_epoch(device, cfg, criterion, pb):
    name = cfg['NAME']
    mode = cfg['MODE']
    ray_hidden_dim = cfg['TRAIN']['Ray_hidden_dim']
    out_dim = cfg['TRAIN']['Out_dim']
    n_tasks = cfg['TRAIN']['N_task']
    num_hidden_layer = cfg['TRAIN']['Solver'][criterion]['Num_hidden_layer']
    last_activation = cfg['TRAIN']['Solver'][criterion]['Last_activation']
    ref_point = tuple(map(int, cfg['TRAIN']['Ref_point'].split(',')))
    lr = cfg['TRAIN']['OPTIMIZER']['Lr']
    wd = cfg['TRAIN']['OPTIMIZER']['WEIGHT_DECAY']
    type_opt = cfg['TRAIN']['OPTIMIZER']['TYPE']
    epochs = cfg['TRAIN']['Epoch']
    alpha_r = cfg['TRAIN']['Alpha']
    start = 0.
    
    if criterion == 'HVI':
        sol = []
        head = cfg['TRAIN']['Solver'][criterion]['Head']
        partition = np.array([1/head]*head)
        hnet = Hypernetwork(ray_hidden_dim = ray_hidden_dim, out_dim = out_dim, n_tasks = n_tasks,num_hidden_layer=num_hidden_layer,last_activation=last_activation)
        hnet = hnet.to(device)
        if type_opt == 'adam':
            optimizer = torch.optim.Adam(hnet.parameters(), lr = lr, weight_decay=wd)
        elif type_opt == 'adamw':
            optimizer = torch.optim.AdamW(hnet.parameters(), lr = lr, weight_decay=wd)

        net_list = []

        n_mo_sol = cfg['TRAIN']['Solver'][criterion]['Head']
        n_mo_obj = cfg['TRAIN']['N_task']
        hesophat = cfg['TRAIN']['Solver'][criterion]['Rho']
        
        end = np.pi/2
        mo_opt = HvMaximization(n_mo_sol, n_mo_obj, ref_point)
        
        dem = 0
        for epoch in tqdm(range(epochs)):
            dem += 1

            hnet.train()
            optimizer.zero_grad()

            loss_torch_per_sample = []
            loss_numpy_per_sample = []
            loss_per_sample = []
            weights = []
            rays = []
            penalty = []
            for i in range(n_mo_sol):
                random = np.random.uniform(start, end)
                if cfg['MODE'] == '2d':
                    ray = torch.from_numpy(
                    np.random.dirichlet((alpha_r, alpha_r), 1).astype(np.float32).flatten()
                    ).to(device)
                else:
                    ray = torch.from_numpy(
                np.random.dirichlet((alpha_r, alpha_r,alpha_r), 1).astype(np.float32).flatten()
                ).to(device)
                rays.append(ray)
                output = hnet(rays[i])
                #output = torch.sqrt(output)
                obj_values = []
                objectives = pb.get_values(output)
                for i in range(len(objectives)):
                    obj_values.append(objectives[i])
                loss_per_sample = torch.stack(obj_values)

                loss_torch_per_sample.append(loss_per_sample)
                loss_numpy_per_sample.append(loss_per_sample.cpu().detach().numpy())

                penalty.append(torch.sum(loss_torch_per_sample[i]*rays[i])/
                                (torch.norm(loss_torch_per_sample[i])*torch.norm(rays[i])))

            loss_numpy_per_sample = np.array(loss_numpy_per_sample)[np.newaxis, :, :].transpose(0, 2, 1) #n_samples, obj, sol

            n_samples = 1
            dynamic_weights_per_sample = torch.ones(n_mo_sol, n_mo_obj, n_samples)
            for i_sample in range(0, n_samples):
                weights_task = mo_opt.compute_weights(loss_numpy_per_sample[i_sample,:,:])
                dynamic_weights_per_sample[:, :, i_sample] = weights_task.permute(1,0)
            
            dynamic_weights_per_sample = dynamic_weights_per_sample.to(device)
            i_mo_sol = 0
            total_dynamic_loss = torch.mean(torch.sum(dynamic_weights_per_sample[i_mo_sol, :, :]
                                                    * loss_torch_per_sample[i_mo_sol], dim=0))
            for i_mo_sol in range(1, len(net_list)):
                total_dynamic_loss += torch.mean(torch.sum(dynamic_weights_per_sample[i_mo_sol, :, :] 
                                                * loss_torch_per_sample[i_mo_sol], dim=0))
            
            for idx in range(head):
                total_dynamic_loss -= hesophat*penalty[idx]*partition[idx]
            
            
            total_dynamic_loss /= head
            total_dynamic_loss.backward()
            optimizer.step()

            penalty = [i.item() for i in penalty]
            sol.append(output.cpu().detach().numpy().tolist()[0])
    else:
        hnet = Hypernetwork(ray_hidden_dim = ray_hidden_dim, out_dim = out_dim, n_tasks = n_tasks,num_hidden_layer=num_hidden_layer,last_activation=last_activation)
        hnet = hnet.to(device)
        sol = []
        if type_opt == 'adam':
            optimizer = torch.optim.Adam(hnet.parameters(), lr = lr, weight_decay=wd) 
        elif type_opt == 'adamw':
            optimizer = torch.optim.AdamW(hnet.parameters(), lr = lr, weight_decay=wd)
        mo_opt = HvMaximization(1, n_tasks, ref_point)
        start = time.time()
        for epoch in tqdm(range(epochs)):
            if cfg['MODE'] == '2d':
                ray = torch.from_numpy(
                np.random.dirichlet((alpha_r, alpha_r), 1).astype(np.float32).flatten()
                ).to(device)
            else:
                ray = torch.from_numpy(
            np.random.dirichlet((alpha_r, alpha_r,alpha_r), 1).astype(np.float32).flatten()
            ).to(device)
            hnet.train()
            optimizer.zero_grad()
            output = hnet(ray)
            output = torch.sqrt(output)
            ray_cs = 1/ray
            ray = ray.squeeze(0)
            obj_values = []
            objectives = pb.get_values(output)
            for i in range(len(objectives)):
                obj_values.append(objectives[i])
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
            sol.append(output.cpu().detach().numpy().tolist()[0])
    end = time.time()
    time_training = end-start
    torch.save(hnet,("./save_weights/best_weight_"+str(criterion)+"_"+str(mode)+"_"+str(name)+"_" + str(cfg['TRAIN']['Ray_hidden_dim'])+".pt"))
    return sol,time_training
