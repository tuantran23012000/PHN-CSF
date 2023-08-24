import logging
import argparse
import json
from collections import defaultdict
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn
from tqdm import trange
import sys
from losses import get_loss
import os
from tqdm import tqdm
print(os.getcwd())
from data import NYUv2
from hyper_segnet import (
    SegNetHyper,
    SegNetTarget,
)
from solvers import EPOSolver, LinearScalarizationSolver, ChebyshevBasedSolver, UtilityBasedSolver
import wandb
import random
from numpy import load
from utils import ConfMatrix, delta_fn, depth_error, normal_error
from pymoo.factory import get_reference_directions
import torch.nn.functional as F
from pymoo.factory import get_performance_indicator
#from pymoo.indicators.hv import HV
import numpy as np

def hypervolumn(A, ref=None, type='acc'):
    """
    :param A: np.array, num_points, num_task
    :param ref: num_task
    """
    dim = A.shape[1]
    if ref is None:
        ref = np.ones(dim)
    hv = get_performance_indicator("hv",ref_point=ref)
    return hv.do(A)

wandb.init()
def get_test_rays(dim, n_partitions):
    """Create 100 rays for evaluation. Not pretty but does the trick"""
    test_rays = get_reference_directions("das-dennis", dim, n_partitions=n_partitions).astype(
        np.float32
    )
    test_rays = test_rays[[(r > 0).all() for r in test_rays]][5:-5:2]
    logging.info(f"initialize {len(test_rays)} test rays")
    return test_rays
def circle_points(K, min_angle=None, max_angle=None):
    # generate evenly distributed preference vector
    ang0 = 1e-6 if min_angle is None else min_angle
    ang1 = np.pi / 2 - ang0 if max_angle is None else max_angle
    angles = np.linspace(ang0, ang1, K, endpoint=True)
    x = np.cos(angles)
    y = np.sin(angles)
    return np.c_[x, y]
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
def calc_loss(x_pred, x_output, task_type,device):
    #device = x_pred.device

    # binary mark to mask out undefined pixel space
    binary_mask = (torch.sum(x_output, dim=1) != 0).float().unsqueeze(1).to(device)

    if task_type == "semantic":
        # semantic loss: depth-wise cross entropy
        loss = F.nll_loss(x_pred, x_output, ignore_index=-1)

    if task_type == "depth":
        # depth loss: l1 norm
        loss = torch.sum(torch.abs(x_pred - x_output) * binary_mask) / torch.nonzero(
            binary_mask, as_tuple=False
        ).size(0)

    if task_type == "normal":
        # normal loss: dot product
        loss = 1 - torch.sum((x_pred * x_output) * binary_mask) / torch.nonzero(
            binary_mask, as_tuple=False
        ).size(0)

    return loss
@torch.no_grad()
def evaluate_hv(hnet, net, loader, rays, device,params):
    hnet.eval()
    with torch.no_grad():  # operations inside don't track history
        losses_all = []
        for ray in tqdm(rays):   
            loss_hv = np.zeros(params["num_tasks"])  
            num_samples = 0
            bs = 0
            ray = torch.from_numpy(ray.astype(np.float32)).to(device)    
            ray /= ray.sum()

            for j, batch in enumerate(loader):
                
                test_data, test_label, test_depth, test_normal = batch
                bs = test_data.shape[0]
                num_samples += bs
                test_data, test_label = test_data.to(device), test_label.long().to(
                    device
                )
                test_depth, test_normal = test_depth.to(device), test_normal.to(device)
                #print(ray)
                
                weights = hnet(ray)
                test_pred= net(test_data,weights)
                test_loss = torch.stack(
                    (
                        calc_loss(test_pred[0], test_label, "semantic",device),
                        calc_loss(test_pred[1], test_depth, "depth",device),
                        #calc_loss(test_pred[2], test_normal, "normal",device),
                    )
                )
                #losses_batch = [test_loss[0].detach().cpu().tolist(),test_loss[1].detach().cpu().tolist(),test_loss[2].detach().cpu().tolist()]
                losses_batch = [test_loss[0].detach().cpu().tolist(),test_loss[1].detach().cpu().tolist()]
                loss_hv += bs * np.array(losses_batch)
            loss_hv /= num_samples
            losses_all.append(loss_hv)


    return np.array(losses_all)
def train(params,device,eval_ray):
    # ----
    # Nets
    # ----

    hn_config = {
        "11M": {"num_chunks": 105, "num_ws": 31},
    }
    hnet = SegNetHyper(hidden_dim=params["hidden_dim"], **hn_config["11M"])
    net = SegNetTarget(n_tasks = params["num_tasks"])

    hnet = hnet.to(device)
    net = net.to(device)
    nyuv2_train_set = NYUv2(
        root="/home/tuantran/pareto-hypernetworks/data/NYUv2", train=True,mode = "train", augmentation=True
    )
    nyuv2_val_set = NYUv2(root="/home/tuantran/pareto-hypernetworks/data/NYUv2", mode = "val",train=False)
    nyuv2_test_set = NYUv2(root="/home/tuantran/pareto-hypernetworks/data/NYUv2", mode = "test",train=False)

    train_loader = torch.utils.data.DataLoader(
        dataset=nyuv2_train_set, batch_size=params["batch_size"], shuffle=True,num_workers=12
    )
    val_loader = torch.utils.data.DataLoader(
        dataset=nyuv2_val_set, batch_size=params["batch_size"],num_workers=12)
    test_loader = torch.utils.data.DataLoader(
        dataset=nyuv2_test_set, batch_size=params["batch_size"], shuffle=False,num_workers=12)

    # ---------
    # Task loss
    # ---------
    num_tasks = params["num_tasks"]
    optimizer = torch.optim.Adam(hnet.parameters(), lr=params["lr"])
    # ------
    # solver
    # ------
    solvers = dict(ls=LinearScalarizationSolver, epo=EPOSolver,cheby=ChebyshevBasedSolver, utility = UtilityBasedSolver)
    solver_type = params["algorithm"]
    solver_method = solvers[solver_type]
    if solver_type == "epo":
        solver = solver_method(n_tasks=num_tasks, n_params=count_parameters(hnet))
    else:
        # ls,cheby,utility
        solver = solver_method(n_tasks=num_tasks)


    start = time.time()
    best_hv = 0
    for epoch in range(params["epochs"]):
        
        sl = []
        dl = []
        nl = []
        al = []
        for j, batch in tqdm(enumerate(train_loader)):
            hnet.train()
            optimizer.zero_grad()
            if alpha > 0:
                ray = torch.from_numpy(
                    np.random.dirichlet(tuple([alpha,alpha]), 1).astype(np.float32).flatten()
                ).to(device)
            weights = hnet(ray)
            train_data, train_label, train_depth, train_normal = batch
            
            train_data, train_label = train_data.to(device), train_label.long().to(
                device
            )
            train_depth, train_normal = train_depth.to(device), train_normal.to(device)

            train_pred = net(train_data, weights)
            #print(train_pred[0].shape,train_label.shape)
            loss_train = torch.stack(
                (
                    calc_loss(train_pred[0], train_label, "semantic",device),
                    calc_loss(train_pred[1], train_depth, "depth",device),
                    #calc_loss(train_pred[2], train_normal, "normal",device),
                )
            )
            loss = solver(loss_train, ray, list(hnet.parameters()))
            loss.backward()
            # print(loss.item())
            # print(loss_train)
            optimizer.step()
            # epoch_iter.set_description(
            #     f"semantic loss: {loss_train[0].item():.3f}, "
            #     f"depth loss: {loss_train[1].item():.3f}, "
            #     f"normal loss: {loss_train[2].item():.3f},"
            #     f"total loss: {loss.item():.3f}"
            # )
            #print(loss.item())
            sl.append(loss_train[0].item())
            dl.append(loss_train[1].item())
            #nl.append(loss_train[2].item())
            al.append(loss.item())

        print("Epoch: ",epoch)
        print("Evaluate")
        loss_eval = evaluate_hv(hnet, net, val_loader, eval_ray, device,params)
        hv_eval = hypervolumn(loss_eval, type='loss', ref=np.ones(params["num_tasks"]) * 3)
        print("HV on eval set: ",hv_eval)
        if hv_eval > best_hv:
            best_hv = hv_eval
            print("Update best model")
            # print("Test")
            # loss_test = evaluate_hv(hnet, net, test_loader, test_ray, device)
            # hv_test = hypervolumn(loss_test, type='loss', ref=np.ones(3) * 3)
            # print("HV on test set: ",hv_test)
            save_dict = {'state_dicts': hnet.state_dict()}
            torch.save(save_dict,"./save_weights/best_model_"+str(params["algorithm"])+"_nyuv2_segnet_mtan_2task.pkl")
        log_dict = {
            "Semantic loss": np.mean(np.array(sl)),
            "Depth loss": np.mean(np.array(dl)),
            #"Normal loss": np.mean(np.array(nl)),
            "HV eval": hv_eval
        }
        wandb.log(log_dict)
    end = time.time()
    print("Training time: ",end-start)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-epochs", type=int, default=150, help="num. epochs")
    parser.add_argument(
        "--ray-hidden", type=int, default=100, help="lower range for ray"
    )
    parser.add_argument("--alpha", type=float, default=0.2, help="alpha for dirichlet")
    parser.add_argument(
        "--model",
        type=str,
        default="lenet",
        choices=["lenet", "resnet"],
        help="model name",
    )
    parser.add_argument(
        "--resnet-size",
        type=str,
        default="11M",
        choices=["11M", "5M", "2M", "1M"],
        help="ResNet size key. Only used if model set to resnet",
    )
    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="train on gpu"
    )
    parser.add_argument("--gpus", type=str, default="0", help="gpu device")
    parser.add_argument("--batch-size", type=int, default=256, help="batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--wd", type=float, default=0.0, help="weight decay")
    parser.add_argument("--val-size", type=float, default=0.1, help="validation size")
    parser.add_argument(
        "--no-val-eval",
        action="store_true",
        default=False,
        help="evaluate on validation",
    )
    parser.add_argument(
        "--solver", type=str, choices=["ls", "epo","cheby","utility"], default="ls", help="solver"
    )
    parser.add_argument(
        "--eval-every",
        type=int,
        default=10,
        help="number of epochs between evaluations",
    )
    parser.add_argument("--out-dir", type=str, default="/home/cist-poc01/tuantran/OPTOPT/pareto-hypernetworks/outputs", help="outputs dir")
    parser.add_argument("--n-rays", type=int, default=25, help="num. rays")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    args = parser.parse_args()

    with open("params.json") as json_params:
        params = json.load(json_params)
    device = torch.device(f"cuda:0" if torch.cuda.is_available() and not False else "cpu")
    alpha = np.random.random(1)[0]
    min_angle = 0.1
    max_angle = np.pi / 2 - 0.1
    eval_ray = circle_points(25, min_angle=min_angle, max_angle=max_angle)
    #eval_ray = get_test_rays(2,12)
    #print(eval_ray)
    print("Number of rays eval: ",eval_ray.shape)
    # test_ray = get_test_rays(2,13)
    # print("Number of rays eval: ",test_ray.shape)
    print("Algrithms: ",params["algorithm"])
    train(params,device,eval_ray)