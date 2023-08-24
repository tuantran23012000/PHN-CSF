import numpy as np
from torch import nn
import torch
from matplotlib import pyplot as plt
from metrics import hypervolumn
import os
from data import Dataset
from models import (
    LeNetHyper,
    LeNetTarget,
)
def evaluate_hv(hypernet, targetnet, loader, rays, device):
    hypernet.eval()
    loss1 = nn.CrossEntropyLoss()
    loss2 = nn.CrossEntropyLoss()
    # num_samples = 0
    losses_all = []
    acc1, acc2 = [], []
    for ray in rays:
        total = 0.0
        task1_correct, task2_correct = 0.0, 0.0
        l1, l2 = 0.0, 0.0
        ray = torch.from_numpy(ray.astype(np.float32)).to(device)
        ray /= ray.sum()
        num_samples = 0
        losses = np.zeros(2)
        for batch in loader:
            hypernet.zero_grad()

            batch = (t.to(device) for t in batch)
            img, ys = batch
            bs = len(ys)
            num_samples += bs
            weights = hypernet(ray)
            logit1, logit2 = targetnet(img, weights)
            curr_l1 = loss1(logit1, ys[:, 0])
            curr_l2 = loss2(logit2, ys[:, 1])
            losses_batch = [curr_l1.detach().cpu().tolist(),curr_l2.detach().cpu().tolist()]
            losses += bs * np.array(losses_batch)
            # acc
            pred1 = logit1.data.max(1)[1]  # first column has actual prob.
            pred2 = logit2.data.max(1)[1]  # first column has actual prob.
            task1_correct += pred1.eq(ys[:, 0]).sum()
            task2_correct += pred2.eq(ys[:, 1]).sum()
        losses /= num_samples
        acc_1 = task1_correct/num_samples
        acc_2 = task2_correct/num_samples
        acc1.append(acc_1.tolist())
        acc2.append(acc_2.tolist())
        losses_all.append(losses.tolist())

    return losses_all, np.array(acc1), np.array(acc2)
def circle_points(K, min_angle=None, max_angle=None):
    # generate evenly distributed preference vector
    ang0 = 1e-6 if min_angle is None else min_angle
    ang1 = np.pi / 2 - ang0 if max_angle is None else max_angle
    angles = np.linspace(ang0, ang1, K, endpoint=True)
    x = np.cos(angles)
    y = np.sin(angles)
    return np.c_[x, y]
def phn_test(dataset,path_data,save_weights,device,mode):
    hidden_dim = 100
    hnet = LeNetHyper([9, 5], ray_hidden_dim=hidden_dim)
    hnet1 = LeNetHyper([9, 5], ray_hidden_dim=hidden_dim)
    hnet2 = LeNetHyper([9, 5], ray_hidden_dim=hidden_dim)
    hnet3 = LeNetHyper([9, 5], ray_hidden_dim=hidden_dim)
    net = LeNetTarget([9, 5])

    # Load dataset
    # MultiMNIST: multi_mnist.pickle
    if dataset == 'mnist':
        path = os.path.join(path_data,'multi_mnist.pickle')

    # MultiFashionMNIST: multi_fashion.pickle
    if dataset == 'fashion':
        path = os.path.join(path_data,'multi_fashion.pickle')

    # Multi-(Fashion+MNIST): multi_fashion_and_mnist.pickle
    if dataset == 'fashion_mnist':
        path = os.path.join(path_data,'multi_fashion_and_mnist.pickle')

    # Load checkpoints
    ckpt = torch.load(os.path.join(save_weights,'best_model_ls_multi_'+str(dataset)+'.pkl'),map_location=device)
    hnet.load_state_dict(ckpt['state_dicts'])
    ckpt1 = torch.load(os.path.join(save_weights,'best_model_utility_multi_'+str(dataset)+'.pkl'),map_location=device)
    hnet1.load_state_dict(ckpt1['state_dicts'])
    ckpt2 = torch.load(os.path.join(save_weights,'best_model_cheby_multi_'+str(dataset)+'.pkl'),map_location=device)
    hnet2.load_state_dict(ckpt2['state_dicts'])
    ckpt3 = torch.load(os.path.join(save_weights,'best_model_epo_multi_'+str(dataset)+'.pkl'),map_location=device)
    hnet3.load_state_dict(ckpt3['state_dicts'])
    hnet = hnet.to(device)
    hnet1 = hnet1.to(device)
    hnet2 = hnet2.to(device)
    hnet3 = hnet3.to(device)

    min_angle = 0.1
    max_angle = np.pi / 2 - 0.1
    num_rays = 25
    test_rays = circle_points(num_rays, min_angle=min_angle, max_angle=max_angle)
    # test_rays = [[0.01,0.99],[0.25,0.75],[0.5,0.5],[0.75,0.25],[0.99,0.01]]
    # test_rays = np.array(test_rays)

    bs = 256
    val_size = 0.1
    data = Dataset(path, val_size=0.1)
    train_set, val_set, test_set = data.get_datasets()
    batch_size = 256
    train_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=batch_size,num_workers=4,
        shuffle=True)
    val_loader = torch.utils.data.DataLoader(
        dataset=val_set,
        batch_size=batch_size,num_workers=4,
        shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        dataset=test_set,
        batch_size=batch_size,num_workers=4,
        shuffle=False)
    loss_hv, acc_task_1_0, acc_task_2_0 = evaluate_hv(
                        hypernet=hnet,
                        targetnet=net,
                        loader=test_loader,
                        rays=test_rays,
                        device=device,
                    )
    loss_hv1, acc_task_1_1, acc_task_2_1 = evaluate_hv(
                        hypernet=hnet1,
                        targetnet=net,
                        loader=test_loader,
                        rays=test_rays,
                        device=device,
                    )
    loss_hv2, acc_task_1_2, acc_task_2_2 = evaluate_hv(
                        hypernet=hnet2,
                        targetnet=net,
                        loader=test_loader,
                        rays=test_rays,
                        device=device,
                    )
    loss_hv3, acc_task_1_3, acc_task_2_3 = evaluate_hv(
                        hypernet=hnet3,
                        targetnet=net,
                        loader=test_loader,
                        rays=test_rays,
                        device=device,
                    )

    '''
        Multi Mnist
        x = [0.2,0.3,0.4,0.5]
        y = [0.2,0.4,0.6,0.8]
        plt.plot([0.2631959812361983,0.2631959812361983],[0.2,0.8],ls='-.',c='black',label='Single task')
        plt.plot([0.2,0.5],[0.33708666510219815,0.33708666510219815],ls='-.',c='black')
    '''
    '''
        Multi Fashion
        x = [0.4,0.6,0.8,1.0]
        y = [0.4,0.6,0.8,1.0]
        plt.plot([0.4857283249686036,0.4857283249686036],[0.4,1],ls='-.',c='black',label='Single task')
        plt.plot([0.4,1],[0.5331778043433081,0.5331778043433081],ls='-.',c='black')
    '''
    '''
        Multi Fashion Mnist
        x = [0.1,0.4,0.7,1.0]
        y = [0.4,0.6,0.8,1.0]
        plt.plot([0.16867540993645222,0.16867540993645222],[0.4,1],ls='-.',c='black',label = 'Single-task')
        plt.plot([0.1,1],[0.44227917699874203,0.44227917699874203],ls='-.',c='black')
    '''
    '''
        # Accuracy MNIST
        x = [0.82,0.84,0.88,0.92]
        y = [0.78,0.82,0.86,0.9]
        plt.plot([0.91,0.91],[0.78,0.9],ls='-.',label = 'Single-task',c='black')
        plt.plot([0.82,0.92],[0.885,0.885],ls='-.',c='black')   
    '''
    '''
        # Accuracy Fashion
        x = [0.68,0.73,0.78,0.83]
        y = [0.6,0.68,0.76,0.84]
        plt.plot([0.82,0.82],[0.6,0.84],ls='-.',label = 'Single-task',c='black')
        plt.plot([0.68,0.83],[0.80,0.80],ls='-.',c='black')
    '''
    '''
        # Accuracy Fashion + MNIST
        x = [0.6,0.72,0.84,0.96]
        y = [0.59,0.68,0.77,0.86]
        plt.plot([0.94,0.94],[0.59,0.86],ls='-.',label = 'Single-task',c='black')
        plt.plot([0.6,0.96],[0.84,0.84],ls='-.',c='black')
    '''
    if mode == "loss":
        if dataset == 'mnist':
            loss_hv = np.array(loss_hv)
            loss_hv1 = np.array(loss_hv1)
            loss_hv2 = np.array(loss_hv2)
            loss_hv3 = np.array(loss_hv3)
            x_ = [loss_hv[0, 0],loss_hv[4, 0],loss_hv[9, 0],loss_hv[14, 0],loss_hv[19, 0],loss_hv[24, 0]]
            y_ = [loss_hv[0, 1],loss_hv[4, 1],loss_hv[9, 1],loss_hv[14, 1],loss_hv[19, 1],loss_hv[24, 1]]
            #plt.plot(loss_hv[:, 0], loss_hv[:, 1],label = 'HPN-LS',marker='*',linestyle = '-')
            plt.plot(loss_hv[:, 0], loss_hv[:, 1],linestyle = '-')
            plt.scatter(x_, y_,marker='*',label = 'PHN-LS')
            x1_ = [loss_hv1[0, 0],loss_hv1[4, 0],loss_hv1[9, 0],loss_hv1[14, 0],loss_hv1[19, 0],loss_hv1[24, 0]]
            y1_ = [loss_hv1[0, 1],loss_hv1[4, 1],loss_hv1[9, 1],loss_hv1[14, 1],loss_hv1[19, 1],loss_hv1[24, 1]]
            plt.plot(loss_hv1[:, 0], loss_hv1[:, 1],linestyle = '-.')
            plt.scatter(x1_, y1_,marker='o',label = 'PHN-Utility')
            x2_ = [loss_hv2[0, 0],loss_hv2[4, 0],loss_hv2[9, 0],loss_hv2[14, 0],loss_hv2[19, 0],loss_hv2[24, 0]]
            y2_ = [loss_hv2[0, 1],loss_hv2[4, 1],loss_hv2[9, 1],loss_hv2[14, 1],loss_hv2[19, 1],loss_hv2[24, 1]]
            plt.plot(loss_hv2[:, 0], loss_hv2[:, 1],linestyle = ':')
            plt.scatter(x2_, y2_,marker='x',label = 'PHN-Cheby')
            x3_ = [loss_hv3[0, 0],loss_hv3[4, 0],loss_hv3[9, 0],loss_hv3[14, 0],loss_hv3[19, 0],loss_hv3[24, 0]]
            y3_ = [loss_hv3[0, 1],loss_hv3[4, 1],loss_hv3[9, 1],loss_hv3[14, 1],loss_hv3[19, 1],loss_hv3[24, 1]]
            plt.plot(loss_hv3[:, 0], loss_hv3[:, 1],linestyle = '--')
            plt.scatter(x3_, y3_,marker='v',label = 'PHN-EPO')
            x = [0.2,0.3,0.4,0.5]
            y = [0.2,0.35,0.5,0.65]
            plt.plot([0.2631959812361983,0.2631959812361983],[0.2,0.65],ls='-.',c='black',label='Single task')
            plt.plot([0.2,0.5],[0.33708666510219815,0.33708666510219815],ls='-.',c='black')
            plt.xlabel("Loss CE task left",fontsize=18)
            plt.ylabel("Loss CE task right",fontsize=18)
            plt.xticks(x)
            plt.yticks(y)
            plt.legend(fontsize=18)
            plt.tight_layout()
            plt.savefig('test_multi_'+str(dataset)+'.jpg')
            plt.savefig('test_multi_'+str(dataset)+'.pdf')
        elif dataset == 'fashion':
            loss_hv = np.array(loss_hv)
            loss_hv1 = np.array(loss_hv1)
            loss_hv2 = np.array(loss_hv2)
            loss_hv3 = np.array(loss_hv3)
            x_ = [loss_hv[0, 0],loss_hv[4, 0],loss_hv[9, 0],loss_hv[14, 0],loss_hv[19, 0],loss_hv[24, 0]]
            y_ = [loss_hv[0, 1],loss_hv[4, 1],loss_hv[9, 1],loss_hv[14, 1],loss_hv[19, 1],loss_hv[24, 1]]
            #plt.plot(loss_hv[:, 0], loss_hv[:, 1],label = 'HPN-LS',marker='*',linestyle = '-')
            plt.plot(loss_hv[:, 0], loss_hv[:, 1],linestyle = '-')
            plt.scatter(x_, y_,marker='*',label = 'PHN-LS')
            x1_ = [loss_hv1[0, 0],loss_hv1[4, 0],loss_hv1[9, 0],loss_hv1[14, 0],loss_hv1[19, 0],loss_hv1[24, 0]]
            y1_ = [loss_hv1[0, 1],loss_hv1[4, 1],loss_hv1[9, 1],loss_hv1[14, 1],loss_hv1[19, 1],loss_hv1[24, 1]]
            plt.plot(loss_hv1[:, 0], loss_hv1[:, 1],linestyle = '-.')
            plt.scatter(x1_, y1_,marker='o',label = 'PHN-Utility')
            x2_ = [loss_hv2[0, 0],loss_hv2[4, 0],loss_hv2[9, 0],loss_hv2[14, 0],loss_hv2[19, 0],loss_hv2[24, 0]]
            y2_ = [loss_hv2[0, 1],loss_hv2[4, 1],loss_hv2[9, 1],loss_hv2[14, 1],loss_hv2[19, 1],loss_hv2[24, 1]]
            plt.plot(loss_hv2[:, 0], loss_hv2[:, 1],linestyle = ':')
            plt.scatter(x2_, y2_,marker='x',label = 'PHN-Cheby')
            x3_ = [loss_hv3[0, 0],loss_hv3[4, 0],loss_hv3[9, 0],loss_hv3[14, 0],loss_hv3[19, 0],loss_hv3[24, 0]]
            y3_ = [loss_hv3[0, 1],loss_hv3[4, 1],loss_hv3[9, 1],loss_hv3[14, 1],loss_hv3[19, 1],loss_hv3[24, 1]]
            plt.plot(loss_hv3[:, 0], loss_hv3[:, 1],linestyle = '--')
            plt.scatter(x3_, y3_,marker='v',label = 'PHN-EPO')
            x = [0.4,0.6,0.8,1.0]
            y = [0.4,0.6,0.8,1.0]
            plt.plot([0.4857283249686036,0.4857283249686036],[0.4,1],ls='-.',c='black',label='Single task')
            plt.plot([0.4,1],[0.5331778043433081,0.5331778043433081],ls='-.',c='black')
            plt.xlabel("Loss CE task left",fontsize=18)
            plt.ylabel("Loss CE task right",fontsize=18)
            plt.xticks(x)
            plt.yticks(y)
            plt.legend(fontsize=18)
            plt.tight_layout()
            plt.savefig('test_multi_'+str(dataset)+'.jpg')
            plt.savefig('test_multi_'+str(dataset)+'.pdf')
        elif dataset == 'fashion_mnist':
            loss_hv = np.array(loss_hv)
            loss_hv1 = np.array(loss_hv1)
            loss_hv2 = np.array(loss_hv2)
            loss_hv3 = np.array(loss_hv3)
            x_ = [loss_hv[0, 0],loss_hv[4, 0],loss_hv[9, 0],loss_hv[14, 0],loss_hv[19, 0],loss_hv[24, 0]]
            y_ = [loss_hv[0, 1],loss_hv[4, 1],loss_hv[9, 1],loss_hv[14, 1],loss_hv[19, 1],loss_hv[24, 1]]
            #plt.plot(loss_hv[:, 0], loss_hv[:, 1],label = 'HPN-LS',marker='*',linestyle = '-')
            plt.plot(loss_hv[:, 0], loss_hv[:, 1],linestyle = '-')
            plt.scatter(x_, y_,marker='*',label = 'PHN-LS')
            x1_ = [loss_hv1[0, 0],loss_hv1[4, 0],loss_hv1[9, 0],loss_hv1[14, 0],loss_hv1[19, 0],loss_hv1[24, 0]]
            y1_ = [loss_hv1[0, 1],loss_hv1[4, 1],loss_hv1[9, 1],loss_hv1[14, 1],loss_hv1[19, 1],loss_hv1[24, 1]]
            plt.plot(loss_hv1[:, 0], loss_hv1[:, 1],linestyle = '-.')
            plt.scatter(x1_, y1_,marker='o',label = 'PHN-Utility')
            x2_ = [loss_hv2[0, 0],loss_hv2[4, 0],loss_hv2[9, 0],loss_hv2[14, 0],loss_hv2[19, 0],loss_hv2[24, 0]]
            y2_ = [loss_hv2[0, 1],loss_hv2[4, 1],loss_hv2[9, 1],loss_hv2[14, 1],loss_hv2[19, 1],loss_hv2[24, 1]]
            plt.plot(loss_hv2[:, 0], loss_hv2[:, 1],linestyle = ':')
            plt.scatter(x2_, y2_,marker='x',label = 'PHN-Cheby')
            x3_ = [loss_hv3[0, 0],loss_hv3[4, 0],loss_hv3[9, 0],loss_hv3[14, 0],loss_hv3[19, 0],loss_hv3[24, 0]]
            y3_ = [loss_hv3[0, 1],loss_hv3[4, 1],loss_hv3[9, 1],loss_hv3[14, 1],loss_hv3[19, 1],loss_hv3[24, 1]]
            plt.plot(loss_hv3[:, 0], loss_hv3[:, 1],linestyle = '--')
            plt.scatter(x3_, y3_,marker='v',label = 'PHN-EPO')
            x = [0.1,0.4,0.7,1.0]
            y = [0.4,0.6,0.8,1.0]
            plt.plot([0.16867540993645222,0.16867540993645222],[0.4,1],ls='-.',c='black',label = 'Single-task')
            plt.plot([0.1,1],[0.44227917699874203,0.44227917699874203],ls='-.',c='black')
            plt.xlabel("Loss CE task left",fontsize=18)
            plt.ylabel("Loss CE task right",fontsize=18)
            plt.xticks(x)
            plt.yticks(y)
            
            plt.legend(fontsize=18)
            plt.tight_layout()
            plt.savefig('test_multi_'+str(dataset)+'.jpg')
            plt.savefig('test_multi_'+str(dataset)+'.pdf')
        print("HV PHN-LS: ",hypervolumn(np.array(loss_hv), type='loss', ref=np.ones(2) * 2))
        print("HV PHN-Utility: ",hypervolumn(np.array(loss_hv1), type='loss', ref=np.ones(2) * 2))
        print("HV PHN-Chebyshev: ",hypervolumn(np.array(loss_hv2), type='loss', ref=np.ones(2) * 2))
        print("HV PHN-EPO: ",hypervolumn(np.array(loss_hv3), type='loss', ref=np.ones(2) * 2))
    else:
        if dataset == 'mnist':
            x_ = [acc_task_1_0[0],acc_task_1_0[4],acc_task_1_0[9],acc_task_1_0[14],acc_task_1_0[19],acc_task_1_0[24]]
            y_ = [acc_task_2_0[0],acc_task_2_0[4],acc_task_2_0[9],acc_task_2_0[14],acc_task_2_0[19],acc_task_2_0[24]]
            plt.plot(acc_task_1_0, acc_task_2_0,linestyle = '-')
            plt.scatter(x_, y_,marker='*',label = 'PHN-LS')
            x1_ = [acc_task_1_1[0],acc_task_1_1[4],acc_task_1_1[9],acc_task_1_1[14],acc_task_1_1[19],acc_task_1_1[24]]
            y1_ = [acc_task_2_1[0],acc_task_2_1[4],acc_task_2_1[9],acc_task_2_1[14],acc_task_2_1[19],acc_task_2_1[24]]
            plt.plot(acc_task_1_1, acc_task_2_1,linestyle = '-.')
            plt.scatter(x1_, y1_,marker='o',label = 'PHN-Utility')
            x2_ = [acc_task_1_2[0],acc_task_1_2[4],acc_task_1_2[9],acc_task_1_2[14],acc_task_1_2[19],acc_task_1_2[24]]
            y2_ = [acc_task_2_2[0],acc_task_2_2[4],acc_task_2_2[9],acc_task_2_2[14],acc_task_2_2[19],acc_task_2_2[24]]
            plt.plot(acc_task_1_2, acc_task_2_2,linestyle = ':')
            plt.scatter(x2_, y2_,marker='x',label = 'PHN-Cheby')
            x3_ = [acc_task_1_3[0],acc_task_1_3[4],acc_task_1_3[9],acc_task_1_3[14],acc_task_1_3[19],acc_task_1_3[24]]
            y3_ = [acc_task_2_3[0],acc_task_2_3[4],acc_task_2_3[9],acc_task_2_3[14],acc_task_2_3[19],acc_task_2_3[24]]
            plt.plot(acc_task_1_3, acc_task_2_3,linestyle = '--')
            plt.scatter(x3_, y3_,marker='v',label = 'PHN-EPO')
            # plt.plot(acc_task_1_1, acc_task_2_1,label = 'HPN-Utility',marker='o',linestyle = '-.')
            # plt.plot(acc_task_1_2, acc_task_2_2,label = 'HPN-Cheby',marker='x',linestyle = ':')
            # plt.plot(acc_task_1_3, acc_task_2_3,label = 'PHN-EPO',marker='v',linestyle = '--')
            x = [0.82,0.84,0.88,0.92]
            y = [0.78,0.82,0.86,0.9]
            plt.plot([0.91,0.91],[0.78,0.9],ls='-.',label = 'Single-task',c='black')
            plt.plot([0.82,0.92],[0.885,0.885],ls='-.',c='black')   
            plt.xlabel("Accuracy task left",fontsize=18)
            plt.ylabel("Accuracytask right",fontsize=18)
            plt.xticks(x)
            plt.yticks(y)
            plt.legend(fontsize=18)
            plt.tight_layout()
            plt.savefig('test_acc_multi_'+str(dataset)+'.jpg')
            plt.savefig('test_acc_multi_'+str(dataset)+'.pdf')
        elif dataset == 'fashion':
            x_ = [acc_task_1_0[0],acc_task_1_0[4],acc_task_1_0[9],acc_task_1_0[14],acc_task_1_0[19],acc_task_1_0[24]]
            y_ = [acc_task_2_0[0],acc_task_2_0[4],acc_task_2_0[9],acc_task_2_0[14],acc_task_2_0[19],acc_task_2_0[24]]
            plt.plot(acc_task_1_0, acc_task_2_0,linestyle = '-')
            plt.scatter(x_, y_,marker='*',label = 'PHN-LS')
            x1_ = [acc_task_1_1[0],acc_task_1_1[4],acc_task_1_1[9],acc_task_1_1[14],acc_task_1_1[19],acc_task_1_1[24]]
            y1_ = [acc_task_2_1[0],acc_task_2_1[4],acc_task_2_1[9],acc_task_2_1[14],acc_task_2_1[19],acc_task_2_1[24]]
            plt.plot(acc_task_1_1, acc_task_2_1,linestyle = '-.')
            plt.scatter(x1_, y1_,marker='o',label = 'PHN-Utility')
            x2_ = [acc_task_1_2[0],acc_task_1_2[4],acc_task_1_2[9],acc_task_1_2[14],acc_task_1_2[19],acc_task_1_2[24]]
            y2_ = [acc_task_2_2[0],acc_task_2_2[4],acc_task_2_2[9],acc_task_2_2[14],acc_task_2_2[19],acc_task_2_2[24]]
            plt.plot(acc_task_1_2, acc_task_2_2,linestyle = ':')
            plt.scatter(x2_, y2_,marker='x',label = 'PHN-Cheby')
            x3_ = [acc_task_1_3[0],acc_task_1_3[4],acc_task_1_3[9],acc_task_1_3[14],acc_task_1_3[19],acc_task_1_3[24]]
            y3_ = [acc_task_2_3[0],acc_task_2_3[4],acc_task_2_3[9],acc_task_2_3[14],acc_task_2_3[19],acc_task_2_3[24]]
            plt.plot(acc_task_1_3, acc_task_2_3,linestyle = '--')
            plt.scatter(x3_, y3_,marker='v',label = 'PHN-EPO')
            x = [0.68,0.73,0.78,0.83]
            y = [0.6,0.68,0.76,0.84]
            plt.plot([0.82,0.82],[0.6,0.84],ls='-.',label = 'Single-task',c='black')
            plt.plot([0.68,0.83],[0.80,0.80],ls='-.',c='black')
            plt.xlabel("Accuracy task left",fontsize=18)
            plt.ylabel("Accuracytask right",fontsize=18)
            plt.xticks(x)
            plt.yticks(y)
            plt.legend(fontsize=18)
            plt.tight_layout()
            plt.savefig('test_acc_multi_'+str(dataset)+'.jpg')
            plt.savefig('test_acc_multi_'+str(dataset)+'.pdf')
        elif dataset == 'fashion_mnist':
            x_ = [acc_task_1_0[0],acc_task_1_0[4],acc_task_1_0[9],acc_task_1_0[14],acc_task_1_0[19],acc_task_1_0[24]]
            y_ = [acc_task_2_0[0],acc_task_2_0[4],acc_task_2_0[9],acc_task_2_0[14],acc_task_2_0[19],acc_task_2_0[24]]
            plt.plot(acc_task_1_0, acc_task_2_0,linestyle = '-')
            plt.scatter(x_, y_,marker='*',label = 'PHN-LS')
            x1_ = [acc_task_1_1[0],acc_task_1_1[4],acc_task_1_1[9],acc_task_1_1[14],acc_task_1_1[19],acc_task_1_1[24]]
            y1_ = [acc_task_2_1[0],acc_task_2_1[4],acc_task_2_1[9],acc_task_2_1[14],acc_task_2_1[19],acc_task_2_1[24]]
            plt.plot(acc_task_1_1, acc_task_2_1,linestyle = '-.')
            plt.scatter(x1_, y1_,marker='o',label = 'PHN-Utility')
            x2_ = [acc_task_1_2[0],acc_task_1_2[4],acc_task_1_2[9],acc_task_1_2[14],acc_task_1_2[19],acc_task_1_2[24]]
            y2_ = [acc_task_2_2[0],acc_task_2_2[4],acc_task_2_2[9],acc_task_2_2[14],acc_task_2_2[19],acc_task_2_2[24]]
            plt.plot(acc_task_1_2, acc_task_2_2,linestyle = ':')
            plt.scatter(x2_, y2_,marker='x',label = 'PHN-Cheby')
            x3_ = [acc_task_1_3[0],acc_task_1_3[4],acc_task_1_3[9],acc_task_1_3[14],acc_task_1_3[19],acc_task_1_3[24]]
            y3_ = [acc_task_2_3[0],acc_task_2_3[4],acc_task_2_3[9],acc_task_2_3[14],acc_task_2_3[19],acc_task_2_3[24]]
            plt.plot(acc_task_1_3, acc_task_2_3,linestyle = '--')
            plt.scatter(x3_, y3_,marker='v',label = 'PHN-EPO')
            x = [0.6,0.72,0.84,0.96]
            y = [0.59,0.68,0.77,0.86]
            plt.plot([0.94,0.94],[0.59,0.86],ls='-.',label = 'Single-task',c='black')
            plt.plot([0.6,0.96],[0.84,0.84],ls='-.',c='black')
            plt.xlabel("Accuracy task left",fontsize=18)
            plt.ylabel("Accuracytask right",fontsize=18)
            plt.xticks(x)
            plt.yticks(y)
            plt.legend(fontsize=18)
            plt.tight_layout()
            plt.savefig('test_acc_multi_'+str(dataset)+'.jpg')
            plt.savefig('test_acc_multi_'+str(dataset)+'.pdf')
def visualize(dataset,path_data,save_weights,device,mode):
    hidden_dim = 100
    # hnet = LeNetHyper([9, 5], ray_hidden_dim=hidden_dim)
    net = LeNetTarget([9, 5])
    #dataset = 'mnist'
    # MultiMNIST: multi_mnist.pickle
    if dataset == 'mnist':
        path = os.path.join(path_data,'multi_mnist.pickle')

    # MultiFashionMNIST: multi_fashion.pickle
    if dataset == 'fashion':
        path = os.path.join(path_data,'multi_fashion.pickle')

    # Multi-(Fashion+MNIST): multi_fashion_and_mnist.pickle
    if dataset == 'fashion_mnist':
        path = os.path.join(path_data,'multi_fashion_and_mnist.pickle')
    # print(path)
    # print(os.path.join(save_weights,'best_model_ls_multi_'+str(dataset)+'.pt'))
    print(torch.load(os.path.join(save_weights,'best_model_ls_multi_'+str(dataset)+'.pt'), map_location=device))
    hnet = torch.load(os.path.join(save_weights,'best_model_ls_multi_'+str(dataset)+'.pt'), map_location=device)
    print(hnet)
    # hnet1 = torch.load(os.path.join(save_weights,'best_model_utility_multi_'+str(dataset)+'.pt'), map_location=device)
    # hnet2 = torch.load(os.path.join(save_weights,'best_model_cheby_multi_'+str(dataset)+'.pt'), map_location=device)
    # hnet3 = torch.load(os.path.join(save_weights,'best_model_epo_multi_'+str(dataset)+'.pt'), map_location=device)


dataset = 'mnist'
path_data = '/home/tuantran/Documents/OPT/Multi_Gradient_Descent/HPN-CSF/MTL/dataset/Multi_task'
save_weights = './save_weights'
device = device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mode = 'loss'
visualize(dataset,path_data,save_weights,device,mode)
