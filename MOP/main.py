import sys
import os
sys.path.append(os.getcwd())
import time
import numpy as np
import torch
from tools.utils import visualize_2d, visualize_3d, visualize_predict_2d, visualize_predict_3d
import argparse
import yaml
from problems.get_problem import Problem
from train import train_epoch
from predict import predict_result
def run_train(cfg,criterion,device,problem,model_type):
    pb = Problem(problem, cfg['MODE'])
    pf = pb.get_pf()
    if cfg['MODE'] == '2d':
        sol, time_training = train_epoch(device,cfg,criterion,pb,pf,model_type)
        print("Time: ",time_training)  
        visualize_2d(sol,pf,cfg,criterion,pb)
    else:
        sol, time_training = train_epoch(device,cfg,criterion,pb,pf,model_type)
        print("Time: ",time_training)  
        visualize_3d(sol,pf,cfg,criterion,pb)
def run_predict(cfg,criterion,device,problem,model_type):
    pb = Problem(problem, cfg['MODE'])
    pf = pb.get_pf()
    if cfg['MODE'] == '2d':
        if cfg['EVAL']['Flag']:
            check = []
            for i in range(cfg['EVAL']['Num_eval']):
                igd, _, _, _,med = predict_result(device,cfg,criterion,pb,pf,model_type,num_e=None,contexts = [])
                check.append(med.tolist())
            print("Mean: ",np.array(check).mean())
            print("Std: ",np.array(check).std())
        else:
            igd, targets_epo, results1, contexts,med = predict_result(device,cfg,criterion,pb,pf,model_type,contexts = [])
            #print(contexts)
            visualize_predict_2d(cfg,targets_epo, results1, contexts,pb,pf,criterion,igd,med)
    else:
        num_ray_init = cfg['EVAL']['Num_ray_init']
        num_ray_test = cfg['EVAL']['Num_ray_test']
        # contexts = get_rays(cfg, num_ray_init)
        # rng = np.random.default_rng()
        # contexts = rng.choice(contexts,num_ray_test)
        # np.save("ray.npy", contexts)
        
        contexts = np.load("ray.npy")
        if cfg['EVAL']['Flag']:
            check = []
            for i in range(cfg['EVAL']['Num_eval']):
                igd, _, _, _, med = predict_result(device,cfg,criterion,pb,pf,model_type,num_e=None,contexts = contexts)
                check.append(med.tolist())
            print("Mean: ",np.array(check).mean())
            print("Std: ",np.array(check).std())
        else:
            igd, targets_epo, results1, contexts,med = predict_result(device,cfg,criterion,pb,pf,model_type,contexts = contexts)
            visualize_predict_3d(cfg,targets_epo, results1, contexts,pb,pf,criterion,igd,med)
if __name__ == "__main__":
    device = torch.device(f"cuda:0" if torch.cuda.is_available() and not False else "cpu")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--solver", type=str, choices=["LS", "KL","Cheby","Utility","Cosine","Cauchy","Prod","Log","AC","MC","HV","CPMTL","EPO","HVI"],
        default="Cheby", help="solver"
    )
    parser.add_argument(
        "--problem", type=str, choices=["ex1", "ex2","ex3","ex4","ZDT1","ZDT2","ZDT3","DTLZ2"],
        default="ex3", help="solver"
    )
    parser.add_argument(
        "--mode", type=str,
        default="test"
    )
    parser.add_argument(
        "--model_type", type=str,
        default="mlp"
    )
    args = parser.parse_args()
    criterion = args.solver 
    print("Scalar funtion: ",criterion)
    problem = args.problem
    config_file = "./configs/"+str(problem)+".yaml"
    model_type = args.model_type
    with open(config_file) as stream:
        cfg = yaml.safe_load(stream)
    if args.mode == "train":
        run_train(cfg,criterion,device,problem,model_type)
    else:
        run_predict(cfg,criterion,device,problem,model_type)