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
import metrics
import os
from tqdm import tqdm
print(os.getcwd())
from data import get_dataset
from hyper_resnet import (
    ResnetHyper,
    ResNetTarget,
)
from pymoo.factory import get_reference_directions
from solvers import EPOSolver, LinearScalarizationSolver, ChebyshevBasedSolver, UtilityBasedSolver
import wandb
import random
from numpy import load
from pymoo.factory import get_performance_indicator
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
def hypervolumn(A, ref=None, type='acc'):
    """
    :param A: np.array, num_points, num_task
    :param ref: num_task
    """
    dim = A.shape[1]

    if type == 'loss':
        if ref is None:
            ref = np.ones(dim)
        hv = get_performance_indicator("hv",ref_point=ref)
        return hv.do(A)
    else:
        print('type not implemented')
        return None
wandb.init()


def expected_calibration_error(y_true, y_pred, num_bins=15):
    pred_y = np.argmax(y_pred, axis=-1)
    correct = (pred_y == y_true).astype(np.float32)
    prob_y = np.max(y_pred, axis=-1)

    b = np.linspace(start=0, stop=1.0, num=num_bins)
    bins = np.digitize(prob_y, bins=b, right=True)

    o = 0
    for b in range(num_bins):
        mask = bins == b
        if np.any(mask):
            o += np.abs(np.sum(correct[mask] - prob_y[mask]))

    return o / y_pred.shape[0]
def evaluate(gt_label,preds_label):
    TP = [0]*len(preds_label[0])
    TN = [0]*len(preds_label[0])
    FP = [0]*len(preds_label[0])
    FN = [0]*len(preds_label[0])
    positive = [0]*len(preds_label[0])
    negative = [0]*len(preds_label[0])

    for i in range(len(preds_label)):
        for j in range(len(preds_label[i])):
            if (gt_label[i][j]==1):
                positive[j] += 1
                if (preds_label[i][j] == gt_label[i][j]): 
                    TP[j] += 1
                else: 
                    FN[j] += 1
            if (gt_label[i][j]==0):
                negative[j] += 1
                if (preds_label[i][j] == gt_label[i][j]):
                    TN[j] += 1
                else:
                    FP[j] += 1

    for i in range(len(TP)):
        if (positive[i] == 0) or (negative[i] == 0):
            continue
        TP[i] /= positive[i]
        FP[i] /= negative[i]
        TN[i] /= negative[i]
        FN[i] /= positive[i]
    TP = np.array(TP)
    TN = np.array(TN)
    FP = np.array(FP)
    FN = np.array(FN)
    label_pos_recall =  np.mean(TP/(TP+FN))   # true positive
    label_neg_recall =  np.mean(TN/(TN+FP))   # true negative
    mA = (label_pos_recall+label_neg_recall)/ 2
    prec = np.mean(TP/(TP+FP))
    recall = np.mean(TP/(TP+FN))
    f1 = np.mean(TP/(TP+(1/2)*(FP+FN)))
    # print("mA: ",mA)
    # print("Precision: ",prec)
    # print("Recall: ",recall)
    # print("F1: ",f1)
    return mA, prec,recall,f1

def evaluate_hv(hnet, net, val_loader, rays, device,params):
    tasks = params["tasks"]
    num_tasks = len(tasks)
    all_tasks = configs[params["dataset"]]["all_tasks"]
    loss_fn = get_loss(params)
    metric = metrics.get_metrics(params)
    hnet.eval()
    loss1 = nn.CrossEntropyLoss()
    loss2 = nn.CrossEntropyLoss()
    # num_samples = 0
    results = defaultdict(list)
    losses_all = []
    accs_all = []
    mA_all = []
    prec_all = []
    re_all = []
    f1_all = []
    for ray in tqdm(rays):

        losses_epoch = []
        acc_epoch = []
        accs = 0
        eces = []
        out_pred = {t: [] for t in tasks}
        out_true = {t: [] for t in tasks}
        num_samples = 0
        losses = np.zeros(40)
        bs = 0
        gt_all = []
        
        pred_all = []
        ray = torch.from_numpy(
            ray.astype(np.float32).flatten()
        ).to(device)
        for k,batch in enumerate(val_loader):
            #hnet.eval()
            hnet.zero_grad()
            img = batch[0].to(device)
            bs = img.shape[0]
            num_samples += bs
            labels = {}
            check = [alpha]*num_tasks
            weights = hnet(ray)
            logit = net(img, weights)
            loss_train = []
            pred_task = []
            gt = []
            for i, t in enumerate(all_tasks):
                # if t not in tasks:
                #     continue
                out_vals = []
                labels = batch[i + 1].to(device)
                out_vals.append(logit[i])
                loss_train.append(loss_fn[t](logit[i],labels))
                out_val = torch.exp(torch.stack(out_vals)).mean(0)
                out_val = torch.log(out_val)
                value,idx = torch.max(out_val,1)
                pred_task.append(idx.detach().cpu().numpy().reshape(1,bs).tolist()[0])
                out_pred[t].append(torch.exp(out_val.data).cpu().numpy())
                out_true[t].append(labels.data.cpu().numpy())
                gt.append(labels.data.cpu().numpy().reshape(1,bs).tolist()[0])
                #print(pred_task)
                metric[t].update(out_val, labels)
            tmp1 = np.array(pred_task).reshape(bs,num_tasks)
            pred_all.append(tmp1)
            tmp2 = np.array(gt).reshape(bs,num_tasks)
            gt_all.append(tmp2)
            #evaluate(tmp2,tmp1)
            losses += bs*np.array(torch.stack(loss_train, -1).detach().cpu().tolist())
        pred_all = np.concatenate(np.array(pred_all,dtype=object),axis = 0)
        gt_all = np.concatenate(np.array(gt_all,dtype=object),axis = 0)
        mA,pre,re,f1 = evaluate(gt_all,pred_all)
        mA_all.append(mA)
        prec_all.append(pre)
        re_all.append(re)
        f1_all.append(f1)
    return np.array(mA_all),np.array(prec_all),np.array(re_all),np.array(f1_all)
def train(params,configs,device,test_ray):
    # ----
    # Nets
    # ----

    hn_config = {
        "resnet18": {"num_chunks": 105, "num_ws": 11,"model_name": "resnet18"},
        "resnet34": {"num_chunks": 105, "num_ws": 21,"model_name": "resnet34"},
        "resnet50": {"num_chunks": 105, "num_ws": 41,"model_name": "resnet50"},
        "resnet101": {"num_chunks": 105, "num_ws": 61,"model_name": "resnet101"},
    }
    hnet = ResnetHyper(hidden_dim=params["hidden_dim"], **hn_config[params["backbone"]],out_dim = 2)
    net = ResNetTarget(model_name = params["backbone"],n_tasks = 40)

    hnet = hnet.to(device)
    net = net.to(device)
    train_loader, train_dst, val_loader, val_dst, test_loader, test_dst = get_dataset(params, configs)
    # ---------
    # Task loss
    # ---------
    loss_fn = get_loss(params)
    metric = metrics.get_metrics(params)
    tasks = params["tasks"]
    num_tasks = len(tasks)
    all_tasks = configs[params["dataset"]]["all_tasks"]
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

    min_angle = 0.1
    max_angle = np.pi / 2 - 0.1
    alpha = params["alpha"]
    # ----------
    # Train loop
    # ----------
    last_eval = -1
    #epoch_iter = trange(params["epochs"])

    val_results = dict()
    test_results = dict()
    best_hv_loss = -1
    start = time.time()
    best_acc = 0
    for epoch in range(params["epochs"]):
        losses_epoch = []
        acc_epoch = []
        accs = 0
        eces = []
        out_pred = {t: [] for t in tasks}
        out_true = {t: [] for t in tasks}
        num_samples = 0
        losses = np.zeros(40)
        bs = 0
        pred_all = []
        gt_all = []
        for k,batch in tqdm(enumerate(train_loader)):
            hnet.train()
            optimizer.zero_grad()
            img = batch[0].to(device)
            labels = {}
            check = [alpha]*num_tasks
            bs = img.shape[0]
            num_samples += bs
            if alpha > 0:
                ray = torch.from_numpy(
                    np.random.dirichlet(tuple(check), 1).astype(np.float32).flatten()
                ).to(device)
            weights = hnet(ray)
            logit = net(img, weights)
            loss_train = []
            pred_task = []
            gt = []
            for i, t in enumerate(all_tasks):
                out_vals = []
                labels = batch[i + 1].to(device)
                out_vals.append(logit[i])
                loss_train.append(loss_fn[t](logit[i],labels))
                out_val = torch.exp(torch.stack(out_vals)).mean(0)
                out_val = torch.log(out_val)
                value,idx = torch.max(out_val,1)
                out_pred[t].append(torch.exp(out_val.data).cpu().numpy())
                out_true[t].append(labels.data.cpu().numpy())
                metric[t].update(out_val, labels)
                pred_task.append(idx.detach().cpu().numpy().reshape(1,bs).tolist()[0])
                gt.append(labels.data.cpu().numpy().reshape(1,bs).tolist()[0])
            tmp1 = np.array(pred_task).reshape(bs,num_tasks)
            pred_all.append(tmp1)
            tmp2 = np.array(gt).reshape(bs,num_tasks)
            gt_all.append(tmp2)
            losses += bs*np.array(torch.stack(loss_train, -1).detach().cpu().tolist())
            ray = ray.squeeze(0)
            loss = solver(torch.stack(loss_train, -1), ray, list(hnet.parameters()))
            loss.backward()
            #print(loss.item())
            losses_epoch.append(loss.item())
            optimizer.step()

        pred_all = np.concatenate(np.array(pred_all,dtype=object),axis = 0)
        gt_all = np.concatenate(np.array(gt_all,dtype=object),axis = 0)
        mA,pre,re,f1 = evaluate(gt_all,pred_all)
        print("Epoch: ", epoch)
        print("Training!")
        print('mA: {:.4f}, Prec: {:.4f}, Rec: {:.4f}, F1: {:.4f}, Loss: {:.4f}'.format(mA,pre, re,f1,np.mean(losses/num_samples)))

        log_dict = {
            "mA": mA,
            "Pre": pre,
            "Recall": re,
            "F1": f1,
            "Loss": np.mean(losses/num_samples)
        }
        wandb.log(log_dict)
        ma_hv,pre_hv,re_hv,f1_hv = evaluate_hv(hnet, net, val_loader, test_ray, device,params)
        print("Evaluating!")
        print('mA: {:.4f}, Prec: {:.4f}, Rec: {:.4f}, F1: {:.4f}'.format(np.mean(ma_hv),np.mean(pre_hv), np.mean(re_hv),np.mean(f1_hv)))
        ma_hv = np.mean(ma_hv)
        if ma_hv > best_acc:
            best_acc = ma_hv
            print("Update best model")
            save_dict = {'state_dicts': hnet.state_dict()}
            torch.save(save_dict,"./save_weights/best_model_"+str(params["algorithm"])+"_celeba_R34.pkl")
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
    parser.add_argument("--n-rays", type=int, default=25, help="num. rays")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    args = parser.parse_args()
    with open("configs.json") as config_params:
        configs = json.load(config_params)

    with open("params.json") as json_params:
        params = json.load(json_params)
    device = torch.device(f"cuda:0" if torch.cuda.is_available() and not False else "cpu")
    alpha = np.random.random(1)[0]
    #test_ray = get_ray_test(alpha,40,5)
    test_ray = load('val_rays.npy')
    train(params,configs,device,test_ray)