from tqdm import trange
from collections import defaultdict
import torch
import numpy as np
import torch.nn.functional as F
from pymoo.factory import get_performance_indicator
import sys
import argparse
from tqdm import trange
from collections import defaultdict
import logging
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from utils import (
    set_seed,
    set_logger,
    count_parameters,
    get_device,
    save_args,
)
from models import HyperNet, TargetNet

from data import get_data
from pymoo.factory import get_reference_directions
from solver import EPOSolver, LinearScalarizationSolver, ChebyshevBasedSolver, UtilityBasedSolver
from scipy.spatial import Delaunay
import time

def get_test_rays():
    """Create 100 rays for evaluation. Not pretty but does the trick"""
    test_rays = get_reference_directions("das-dennis", 7, n_partitions=11).astype(
        np.float32
    )
    test_rays = test_rays[[(r > 0).all() for r in test_rays]][5:-5:2]
    logging.info(f"initialize {len(test_rays)} test rays")
    return test_rays

def sample(n,i,unit,vector,rays_vector):
    if 1-sum(vector[:i]) < 0:
      pass
    else:
      if i == n-1:
          vector[i] = 1-sum(vector[:i])
          """if min(vector) <0.05 or max(vector)>0.95:
            pass
          else:"""
          rays_vector.append(vector.copy())
          return vector
      for value in unit:
          vector[i] = value
          sample(n,i+1,unit,vector,rays_vector)

@torch.no_grad()
def evaluate(hypernet, targetnet, loader, rays, device,epoch,name,n_tasks):
    hypernet.eval()
    results = defaultdict(list)
    loss_total = None
    #front = []
    for ray in rays:
        ray = torch.from_numpy(ray.astype(np.float32)).to(device)

        ray /= ray.sum()

        total = 0.0
        full_losses = []
        for batch in loader:
            hypernet.zero_grad()

            batch = (t.to(device) for t in batch)
            xs, ys = batch
            bs = len(ys)

            weights = hypernet(ray)
            pred = targetnet(xs, weights)

            # loss
            curr_losses = get_losses(pred, ys)
            # metrics
            ray = ray.squeeze(0)

            # losses
            full_losses.append(curr_losses.detach().cpu().numpy())
            total += bs
        if loss_total is None:
            loss_total = np.array(np.array(full_losses).mean(0).tolist(),dtype='float32')
        else:
            loss_total += np.array(np.array(full_losses).mean(0).tolist(),dtype='float32')
        results["ray"].append(ray.cpu().numpy().tolist())
        results["loss"].append(np.array(full_losses).mean(0).tolist())
    print("\n")
    print(str(name)+" losses at "+str(epoch)+":",loss_total/len(rays))
    hv = get_performance_indicator(
        "hv",
        ref_point=np.ones(
            n_tasks,
        ),
    )
    hv_result = hv.do(np.array(results["loss"]))
    results["hv"] = hv_result

    return results



# ---------
# Task loss
# ---------
def get_losses(pred, label):
    return F.mse_loss(pred, label, reduction="none").mean(0)

def train_PHN(
    method,
    hnet,
    net,
    optimizer,
    device,
    alpha,
    train_loader,
    val_loader,
    test_loader,
    epochs,
    test_rays,
    eval_every,
    clip,
    n_tasks
):

    if method == "epo":
        solver = EPOSolver(n_tasks=2, n_params=count_parameters(hnet))
    elif method == "ls":
        # ls,cheby,utility
        solver = LinearScalarizationSolver()
    elif method == "cheby":
        # ls,cheby,utility
        solver = ChebyshevBasedSolver(lower_bound = 0.0001)
    elif method == "utility":
        # ls,cheby,utility
        solver = UtilityBasedSolver(upper_bound = 70.0)
    else:
        raise Exception(f"Solver method: {method} not support")


    test_hv = -1
    val_hv = -1
    early_stop = 0
    patience = 0
    epoch_iter = trange(epochs)
    start = time.time()
    for epoch in epoch_iter:
        loss_per_e = None

        if early_stop == 100:
            print("Early stop.")
            break
        # if (early_stop+1) % 16 == 0:
        #     hesophat *= 0.5
        
        if (patience+1) % 40 == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= np.sqrt(0.5)
            patience = 0
            #lr *= np.sqrt(0.5)

        for i, batch in enumerate(train_loader):
            hnet.train()
            optimizer.zero_grad()
            batch = (t.to(device) for t in batch)
            xs, ys = batch

            ray = torch.from_numpy(
                np.random.dirichlet([alpha] * 7, 1).astype(np.float32).flatten()
            ).to(device)

            weights = hnet(ray)
            pred = net(xs, weights)

            losses = get_losses(pred, ys)

            ray = ray.squeeze(0)
            # TODO: ham vo huong hoa
            loss = solver(losses, ray, list(hnet.parameters()))
            #print(losses.data)
            loss.backward()
            epoch_iter.set_description(
                f"total weighted loss: {loss.item():.3f}, MSE: {losses.mean().item():.3f}, "
                f"val HV {val_hv:.4f}"
            )
            if loss_per_e is None:
                loss_per_e = (losses.cpu().detach().numpy()*len(ys))
            else:
                loss_per_e += (losses.cpu().detach().numpy()*len(ys))
            # grad clip
            if clip > 0:
                torch.nn.utils.clip_grad_norm_(hnet.parameters(), clip)

            optimizer.step()
        print("\n")
        print("Training losses per epoch:",loss_per_e/len(train_loader.dataset))
        # if (epoch + 1) % eval_every == 0:
        #     last_eval = epoch
        val_epoch_results = evaluate(
            hypernet=hnet,
            targetnet=net,
            loader=val_loader,
            rays=test_rays,
            device=device,
            name = "Val",
            epoch = epoch,
            n_tasks = n_tasks,
        )

        # TODO: Cho nay de save best model
        if val_epoch_results["hv"] > val_hv:
            val_hv = val_epoch_results["hv"]
            torch.save(hnet,'./outputs/SARCOS_'+ str(method)  + '_best.pt')
            print("Update best weight")
    hnet = torch.load('/home/tuantran/pareto-hypernetworks/outputs/SARCOS_'+ str(method)  + '_best.pt')
    end = time.time()
    print("Training time: ",end-start)
    test_epoch_results = evaluate(
        hypernet=hnet,
        targetnet=net,
        loader=test_loader,
        rays=test_rays,
        device=device,
        name = "Test",
        epoch = epoch,
        n_tasks = n_tasks,
    )
    print("HV on test:",test_epoch_results["hv"])
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SARCOS")

    parser.add_argument("--datapath", type=str, default="data", help="path to data")
    parser.add_argument("--n-epochs", type=int, default=1000, help="num. epochs")
    parser.add_argument(
        "--ray-hidden", type=int, default=25, help="lower range for ray"
    )
    parser.add_argument("--alpha", type=float, default=0.2, help="alpha for dirichlet")
    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="train on gpu"
    )
    parser.add_argument("--gpus", type=str, default="0", help="gpu device")
    parser.add_argument(
        "--optim",
        type=str,
        default="adam",
        choices=["adam", "sgd"],
        help="optimizer type",
    )
    parser.add_argument("--batch-size", type=int, default=2048, help="batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--wd", type=float, default=0.0, help="weight decay")
    parser.add_argument("--val-size", type=float, default=0.1, help="validation size")
    parser.add_argument(
        "--no-val-eval",
        action="store_true",
        default=True,
        help="evaluate on validation",
    )
    parser.add_argument("--clip", type=float, default=-1, help="grad clipping")

    parser.add_argument(
        "--eval-every",
        type=int,
        default=1,
        help="number of epochs between evaluations",
    )

    parser.add_argument("--out-dir", type=str, default="output", help="outputs dir")
    parser.add_argument("--seed", type=int, default=42, help="random seed")

    #parser.add_argument("--choice", type=str, default="partition", help="partition")

    parser.add_argument("--n-mo-sol", type=int, default=32, help="random seed")

    parser.add_argument("--hesophat", type=float, default=0.1, help="penanty paramter")

    parser.add_argument('--solver', type=str, choices=['ls', 'epo', 'cheby', 'utility'], default='cheby', help='solver')

    args = parser.parse_args()
    

    # if args.choice == "partition":
    #     args.choice = True
    # else:
    #     args.choice = False

    set_seed(args.seed)
    set_logger()

    n_mo_obj = 7
    ref_point = [1]*n_mo_obj
    n_tasks = n_mo_obj
    head = args.n_mo_sol
    device=get_device(no_cuda=args.no_cuda, gpus=args.gpus)

    # solver = MultiHead(args.n_mo_sol, n_mo_obj, ref_point)
    hnet: nn.Module = HyperNet()
    net: nn.Module = TargetNet()
    hnet = hnet.to(device)
    net = net.to(device)
    optimizer = torch.optim.Adam(hnet.parameters(), lr=args.lr, weight_decay=0.)
    optimizer_direction = torch.optim.Adam(hnet.parameters(), lr=1e-4, weight_decay=0.)

    train_set, val_set, test_set = get_data(args.datapath)

    bs = args.batch_size

    train_loader = torch.utils.data.DataLoader(
        dataset=train_set, batch_size=bs, shuffle=True, num_workers=0
    )
    val_loader = torch.utils.data.DataLoader(
        dataset=val_set, batch_size=bs, shuffle=True, num_workers=0
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_set, batch_size=bs, shuffle=False, num_workers=0
    )


    test_rays = get_test_rays()

    unit = np.linspace(0, 1, 3)
    vector = [0]*n_tasks
    rays_vector = []
    sample(n_tasks,0,unit,vector,rays_vector)
    rays_sampling = np.array(rays_vector)
    rays_vector_2 = np.delete(rays_sampling, -1, axis=1)
    tri = Delaunay(rays_vector_2)
    partition = tri.simplices
    partition = partition.astype(np.int32)
    #n_mo_sol = tri.simplices.shape[0]
    hesophat = args.hesophat
    head = args.n_mo_sol
    
    method = args.method

    sys.stdout = open('./log_journal/SARCOS_'+ str(method)  + '_log.txt', 'w')

    print("Start training!")
    start = time.time()
    train_PHN(
        method,
        hnet,
        net,
        optimizer,
        device,
        args.alpha,
        train_loader,
        val_loader,
        test_loader,
        args.n_epochs,
        test_rays,
        args.eval_every,
        args.clip,
        n_tasks
        )
    save_args(folder=args.out_dir, args=args)

    print("Running time:", time.time() - start)
    sys.stdout.close()