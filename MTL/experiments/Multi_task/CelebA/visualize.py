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
import cv2
from tqdm import tqdm
from matplotlib import pyplot as plt
print(os.getcwd())
from data import get_dataset
from hyper_resnet import (
    ResnetHyper,
    ResNetTarget,
)
from pymoo.factory import get_reference_directions
from solvers import EPOSolver, LinearScalarizationSolver, ChebyshevBasedSolver, UtilityBasedSolver
import random
from typing import List
import torchvision.transforms as T
import torch
import torch.nn.functional as F
from torch import nn
import torchvision.models as models
from pymoo.factory import get_performance_indicator
import shap
import scipy.misc as m



class Target(nn.Module):
    def __init__(self, pretrained=False, progress=True, weights=None,idx=None, **kwargs):
        super().__init__()
        self.resnet = models.resnet18(
            pretrained=pretrained, progress=progress, num_classes=512, **kwargs
        )

        self.resnet.conv1.weight.data = torch.randn((64, 1, 7, 7))

        self.task1 = nn.Linear(512, 10)
        self.task2 = nn.Linear(512, 10)
        self.weights = weights
        self.idx = idx
    def forward(self, x):
        # pad input
        idx = self.idx
        x = F.pad(input=x, pad=[0, 2, 0, 2], mode="constant", value=0.0)
        weights = self.weights
        if weights is None:
            x = self.resnet(x)
            x = F.relu(x)
            p1, p2 = self.task1(x), self.task2(x)
            return p1, p2

        else:
            x = self.forward_init(x, weights)
            x = self.forward_layer(x, weights, 1)
            x = self.forward_layer(x, weights, 2)
            x = self.forward_layer(x, weights, 3)
            x = self.forward_layer(x, weights, 4)
            #print(x.shape)
            # x = F.adaptive_avg_pool2d(x, (1, 1))
            # print(x.shape)
            # x = torch.flatten(x, 1)
            x = F.avg_pool2d(x, 4)
            #print(x.shape)
            x = x.view(x.size(0), -1)
            out = []
            for i in range(40):
                x1 = self.forward_clf(x, weights, i+1)
                x1 = F.softmax(x1, dim=1)
                # value,i = torch.max(x1,1)
                out.append(x1)

            return out[idx]

    @staticmethod
    def forward_init(x, weights):
        """Before blocks"""
        device = x.device
        x = F.conv2d(x, weights["resnet.conv1.weight"], stride=2, padding=3)
        x = F.batch_norm(
            x,
            torch.zeros(x.data.size()[1]).to(device),
            torch.ones(x.data.size()[1]).to(device),
            weights["resnet.bn1.weight"],
            weights["resnet.bn1.bias"],
            training=True,
        )
        x = F.relu(x)
        #x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1, dilation=1)

        return x

    def forward_block(self, x, weights, layer, index):
        if layer == 1:
            stride = 1
        else:
            stride = 2 if index == 0 else 1

        device = x.device
        identity = x

        # conv
        out = F.conv2d(
            x,
            weights[f"resnet.layer{layer}.{index}.conv1.weight"],
            stride=stride,
            padding=1,
        )
        # bn
        out = F.batch_norm(
            out,
            torch.zeros(out.data.size()[1]).to(device),
            torch.ones(out.data.size()[1]).to(device),
            weights[f"resnet.layer{layer}.{index}.bn1.weight"],
            weights[f"resnet.layer{layer}.{index}.bn1.bias"],
            training=True,
        )
        out = F.relu(out, inplace=True)
        # conv
        out = F.conv2d(
            out,
            weights[f"resnet.layer{layer}.{index}.conv2.weight"],
            stride=1,
            padding=1,
        )
        # bn
        out = F.batch_norm(
            out,
            torch.zeros(out.data.size()[1]).to(device),
            torch.ones(out.data.size()[1]).to(device),
            weights[f"resnet.layer{layer}.{index}.bn2.weight"],
            weights[f"resnet.layer{layer}.{index}.bn2.bias"],
            training=True,
        )

        if layer > 1 and index == 0:
            identity = self.forward_dowmsample(x, weights, layer)

        out += identity

        out = F.relu(out)

        return out

    @staticmethod
    def forward_dowmsample(x, weights, layer):
        device = x.device

        out = F.conv2d(
            x, weights[f"resnet.layer{layer}.0.downsample.0.weight"], stride=2
        )

        out = F.batch_norm(
            out,
            torch.zeros(out.data.size()[1]).to(device),
            torch.ones(out.data.size()[1]).to(device),
            weights[f"resnet.layer{layer}.0.downsample.1.weight"],
            weights[f"resnet.layer{layer}.0.downsample.1.bias"],
            training=True,
        )
        return out

    def forward_layer(self, x, weights, layer):
        x = self.forward_block(x, weights, layer, 0)
        x = self.forward_block(x, weights, layer, 1)
        return x

    @staticmethod
    def forward_linear(x, weights):
        return F.linear(x, weights["resnet.fc.weight"], weights["resnet.fc.bias"])

    @staticmethod
    def forward_clf(x, weights, index):
        return F.linear(x, weights[f"task{index}.weight"], weights[f"task{index}.bias"])
def get_last_conv(m):
    convs = filter(lambda k: isinstance(k, torch.nn.Conv2d), m.modules())
    return list(convs)[-1]
def transform_img(img):
    """transform
    Mean substraction, remap to [0,1], channel order transpose to make Torch happy
    """
    img = img[...,:3]
    img = img.astype(np.float64)
    print(img.shape)
    mean = np.array([73.15835921, 82.90891754, 72.39239876])
    img -= mean
    img = m.imresize(img, (64, 64))
    # Resize scales images from 0 to 255, thus we need
    # to divide by 255.0
    img = img.astype(float) / 255.0
    # NHWC -> NCWH
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).float()
    return img

class Grad_Cam:
    def __init__(self, model,target_layer_names, use_cuda):
        self.model = model
        self.target = target_layer_names
        self.use_cuda = use_cuda
        self.grad_val = []
        self.feature = [] 
        self.hook = []
        self.img = []
        self.inputs = None
        self._register_hook()
    def get_grad(self,module,input,output):
            self.grad_val.append(output[0].detach())
    def get_feature(self,module,input,output):
            self.feature.append(output.detach())
    def _register_hook(self):
        for i in self.target:
                self.hook.append(i.register_forward_hook(self.get_feature))
                self.hook.append(i.register_full_backward_hook(self.get_grad))

    def _normalize(self,cam):
        h,w,c = self.inputs.shape
        cam = (cam-np.min(cam))/np.max(cam)
        cam = cv2.resize(cam, (w,h))

        heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255

        cam = heatmap + np.float32(self.inputs)
        cam = cam / np.max(cam)
        return np.uint8(255*cam)

    def remove_hook(self):
        for i in self.hook:
            i.remove()

    def _preprocess_image(self,img):
         means = [0.485, 0.456, 0.406]
         stds = [0.229, 0.224, 0.225]

         preprocessed_img = img.copy()[:, :, ::-1]
         for i in range(3):
             preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
             preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]
         preprocessed_img = \
         np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))
         preprocessed_img = torch.from_numpy(preprocessed_img)
         preprocessed_img.unsqueeze_(0)
         input = preprocessed_img.requires_grad_(True)
         return input

    def __call__(self, img,idx=None):
        
        self.model.zero_grad()
        self.inputs = np.float32(cv2.resize(img, (64, 64))) / 255
        inputs = self._preprocess_image(self.inputs)
        if self.use_cuda:
            inputs = inputs.cuda()
            #inputs = inputs.unsqueeze(0)
            print(inputs.shape)
            self.model = self.model.cuda()
        output = self.model(inputs)
        if idx is None:
            idx = np.argmax(output.detach().cpu().numpy()) 
        print(output)
        print(idx)
        target = output[0][idx]
        print(target)
        target.backward()
        #computer 
        weights = []
        print(self.grad_val)
        for i in self.grad_val[::-1]: 
             weights.append(np.mean(i.squeeze().cpu().numpy(),axis=(1,2)))
        for index,j in enumerate(self.feature):
             cam = (j.squeeze().cpu().numpy()*weights[index][:,np.newaxis,np.newaxis]).sum(axis=0)
             cam = np.maximum(cam,0) # relu
             self.img.append(self._normalize(cam))
        return self.img
def visualize(params,configs,device):
    hn_config = {
        "resnet18": {"num_chunks": 105, "num_ws": 11,"model_name": "resnet18"},
        "resnet34": {"num_chunks": 105, "num_ws": 21,"model_name": "resnet34"},
        "resnet50": {"num_chunks": 105, "num_ws": 41,"model_name": "resnet50"},
        "resnet101": {"num_chunks": 105, "num_ws": 61,"model_name": "resnet101"},
    }
    hnet = ResnetHyper(hidden_dim=params["hidden_dim"], **hn_config[params["backbone"]],out_dim = 2)
    net = ResNetTarget(model_name = params["backbone"],n_tasks = 40)
    ckpt = torch.load("./save_weights/test_utility.pkl",map_location=device)
    hnet.load_state_dict(ckpt['state_dicts'])
    hnet = hnet.to(device)
    net = net.to(device)
    train_loader, train_dst, val_loader, val_dst, test_loader, test_dst = get_dataset(params, configs)
    tasks = params["tasks"]
    num_tasks = len(tasks)
    all_tasks = configs[params["dataset"]]["all_tasks"]
    check = [0.6]*40
    ray = torch.from_numpy(
            np.random.dirichlet(tuple(check), 1).astype(np.float32).flatten()
        ).to(device)
    flag = False
    # pred = None
    # gt = None
    count = 0
    weights = None
    checkkkkk = None
    ground_truth = None
    predict = None
    img_path_all = None
    from tqdm import tqdm
    id = None
    idxe = 22
    for k,(batch,img_path) in tqdm(enumerate(test_loader)):
        #print(img_path)
        hnet.eval()
        net.eval()
        # hnet.zero_grad()
        with torch.no_grad():
            img = batch[0].to(device)
            bs = img.shape[0]
            weights = hnet(ray)
            logit = net(img, weights)
            #print(weights)
            gt = []
            pred = []
            out_vals = []
            check = 0
            tmp1 = None
            tmp2 = None
            for i, t in enumerate(all_tasks):
                out_vals = []
                labels = batch[i + 1].to(device)
                idx = logit[i].max(1)[1]
                gt.append(labels.data.cpu().numpy().reshape(1,bs).tolist()[0])
                pred.append(idx.detach().cpu().numpy().reshape(1,bs).tolist()[0])
            tmp2 = np.array(gt).T
            tmp1 = np.array(pred).T
            for k in range(bs):
                #print(tmp2[k,idx])
                if tmp2[k,idxe] == 1:
                    if tmp1[k,idxe] == tmp2[k,idxe]:
                        print(k)
                        id = k
                        img_path_all = img_path
                        checkkkkk = img
                        ground_truth = tmp2
                        predict = tmp1
                        flag = True
                        break
            if flag:
                break
    from PIL import Image
    import matplotlib.pyplot
    import skimage.transform
    img_show = []
    for img_path in img_path_all:
        # img = matplotlib.pyplot.imread(img_path)
        image = Image.open(img_path)
        #print(img.shape)
        img = image.resize((64,64))
        # print(img.size)
        # print(img)
        print(np.asarray(img).shape)
        img_show.append(np.asarray(img))
        # print(img.shape)
        # print(type(img))
        #image.show()
    print(np.array(img_show).shape)
    #print(np.array(img_show))
    
    test_images = checkkkkk
    batch,_ = next(iter(val_loader))
    images = batch[0]
    images = images.to(device)
    background = images[0:4]
    net = Target(weights = weights,idx = idxe).to(device)
    net.eval()
    #masker = shap.maskers.Image(images[0].shape)
    pred = net(test_images)
    print(pred)
    e = shap.GradientExplainer(net,background)
    shap_values = e.shap_values(test_images)
    shap_numpy = [np.swapaxes(np.swapaxes(s, 1, -1), 1,2) for s in shap_values][0][id,:]
    #print(shap_numpy.shape)
    #print(test_images.detach().cpu().numpy().shape)
    test_numpy = (np.swapaxes(np.swapaxes(test_images.detach().cpu().numpy(), 1, -1),1,2)*255.0).astype(np.uint8)
    print(test_numpy.shape)
    shap.image_plot(shap_numpy, (-(np.array(img_show)[id,:])*255.0).astype(np.uint8),labels = ['0'],show=False)
    plt.savefig('check.jpg')
    # m = get_last_conv(net)
    # target_layer = [m]
    # Grad_cams = Grad_Cam(net,target_layer,torch.cuda.is_available())
    # img = cv2.imread(img_path_all, 1)
    # print(img.shape)
    # grad_cam_list  = Grad_cams(img)
    # print(grad_cam_list)
    # cv2.imwrite("out.jpg",grad_cam_list[0])
    
    # # ----
    # # Nets
    # # ----
    # hn_config = {
    #     "11M": {"num_chunks": 105, "num_ws": 11},
    # }
    # hnet = ResnetHyper(hidden_dim=params["hidden_dim"], **hn_config["11M"],out_dim = 2)
    # ckpt = torch.load("/home/tuantran/pareto-hypernetworks/experiments/celebA/save_weights/test_utility.pkl",map_location=device)
    # hnet.load_state_dict(ckpt['state_dicts'])
    # #net = net.to(device)
    # hnet = hnet.to(device)
    # train_loader, train_dst, val_loader, val_dst, test_loader, test_dst = get_dataset(params, configs)
    # #ray = [0.001]*40
    # idx = 36 #6:lip,15:eyeglass,35:hat,32:hair
    # #ray[idx] = 0.961
    # #ray = np.array(ray)
    # num_samples = 0
    # batch,_ = next(iter(test_loader))
    # images = batch[0]
    # #print(images.shape)
    # #print(ys.shape)
    # #images = images.view(-1, 1, 36, 36)
    # images = images.to(device)
    # #images /= 255
    # all_tasks = configs[params["dataset"]]["all_tasks"]
    # l = []
    # for i, t in enumerate(all_tasks):
    #     labels = batch[i + 1].to(device)
    #     l.append(labels[250].item())
    # background = images[0:250]
    # from PIL import Image
  
    # # open method used to open different extension image file
    # # import torchvision.transforms as T
    # # train_transform = T.Compose([
    # #     #T.Resize((height, width)),
    # #     #T.Pad(10),
    # #     #T.RandomGrayscale(),
    # #     #Pad(w=width, h=height),
    # #     #T.Pad([Pad_(w=width, h=height)]),
    # #     T.Resize((64, 64)),
    # #     T.ToTensor(),
    # #     #T.RandomErasing(),
    # # ])
    # # import scipy.misc as m
    # # img_path = "/home/tuantran/pareto-hypernetworks/experiments/celebA/big_lips.png"
    # # im = m.imread(img_path)
    # # test_images = transform_img(im).unsqueeze(0).to(device)
    # # print(test_images.shape)
    # test_images = images[250:253]
    # print(l)
    # print(np.array(l)[:][250])
    # print(batch[i + 1])
    # hnet.eval()
    # hnet.zero_grad()
    # #ray = torch.from_numpy(ray.astype(np.float32)).to(device)
    # check = [0.6]*40
    # ray = torch.from_numpy(
    #         np.random.dirichlet(tuple(check), 1).astype(np.float32).flatten()
    #     ).to(device)
    # #print(ray)
    # weights = hnet(ray)
    # net = ResNetTarget(weights = weights,idx = idx).to(device)
    # #masker = shap.maskers.Image(images[0].shape)
    # pred = net(test_images)
    # print(pred)
    # e = shap.GradientExplainer(net,background)
    # shap_values = e.shap_values(test_images)
    # shap_numpy = [np.swapaxes(np.swapaxes(s, 1, -1), 1,2) for s in shap_values]
    # #print(test_images.detach().cpu().numpy().shape)
    # test_numpy = (np.swapaxes(np.swapaxes(test_images.detach().cpu().numpy(), 1, -1),1,2)*255.0).astype(np.uint8)

    # shap.image_plot(shap_numpy, -test_numpy,show=False)

    # plt.savefig('eyeglass.jpg')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MultiFashion")
    parser.add_argument(
        "--datapath",
        type=str,
        default="/home/tuantran/pareto-hypernetworks/data/ParetoMTL_multiMNIST/multi_fashion.pickle",
        help="path to data",
    )
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

    # set_seed(args.seed)
    # set_logger()
    with open("configs.json") as config_params:
        configs = json.load(config_params)

    with open("params.json") as json_params:
        params = json.load(json_params)
    device = torch.device(f"cuda:0" if torch.cuda.is_available() and not False else "cpu")
    alpha = np.random.random(1)[0]
    print(device)
    visualize(params,configs,device)