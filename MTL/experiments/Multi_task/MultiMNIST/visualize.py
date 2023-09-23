import shap
import numpy as np
from torch import nn
import torch
from matplotlib import pyplot as plt
import torch.nn.functional as F
from metrics import hypervolumn
import os
import sys
#sys.path.append("/home/tuantran/pareto-hypernetworks")
from data import Dataset
import torchvision.transforms as transforms
import matplotlib as mpl
mpl.rcParams['xtick.labelsize'] = 15
mpl.rcParams['ytick.labelsize'] = 15
#plt.rcParams["figure.figsize"] = (30,50)

class LeNetTarget1(nn.Module):
    """LeNet target network"""

    def __init__(
        self,
        kernel_size,
        n_kernels=10,
        out_dim=10,
        target_hidden_dim=50,
        n_conv_layers=2,
        n_tasks=2,
        weights=None,
    ):
        super().__init__()
        assert len(kernel_size) == n_conv_layers, (
            "kernel_size is list with same dim as number of "
            "conv layers holding kernel size for each conv layer"
        )
        self.n_kernels = n_kernels
        self.kernel_size = kernel_size
        self.out_dim = out_dim
        self.n_conv_layers = n_conv_layers
        self.n_tasks = n_tasks
        self.target_hidden_dim = target_hidden_dim
        self.weights = weights
    def forward(self, x):
        weights = self.weights 
        #print(x.shape)
        #x = x[0,:,:,:]
        x = F.conv2d(
            x,
            weight=weights["conv0.weights"].reshape(
                self.n_kernels, 1, self.kernel_size[0], self.kernel_size[0]
            ),
            bias=weights["conv0.bias"],
            stride=1,
        )
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        for i in range(1, self.n_conv_layers):
            x = F.conv2d(
                x,
                weight=weights[f"conv{i}.weights"].reshape(
                    int(2 ** i * self.n_kernels),
                    int(2 ** (i - 1) * self.n_kernels),
                    self.kernel_size[i],
                    self.kernel_size[i],
                ),
                bias=weights[f"conv{i}.bias"],
                stride=1,
            )
            x = F.relu(x)
            x = F.max_pool2d(x, 2)

        x = torch.flatten(x, 1)

        x = F.linear(
            x,
            weight=weights["hidden0.weights"].reshape(
                self.target_hidden_dim, x.shape[-1]
            ),
            bias=weights["hidden0.bias"],
        )

        logits = []
        for j in range(self.n_tasks):
            # logits.append(
            #     F.linear(
            #         x,
            #         weight=weights[f"task{j}.weights"].reshape(
            #             self.out_dim, self.target_hidden_dim
            #         ),
            #         bias=weights[f"task{j}.bias"],
            #     )
            # )
            x1 = F.linear(
                    x,
                    weight=weights[f"task{j}.weights"].reshape(
                        self.out_dim, self.target_hidden_dim
                    ),
                    bias=weights[f"task{j}.bias"],
                )
            x2= F.softmax(x1,dim=1)
            logits.append(x2)
        return logits[1]
def visualize(dataset,path_data,save_weights,device,mode):
    hidden_dim = 100
    
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
    hnet = torch.load(os.path.join(save_weights,'best_model_ls_multi_'+str(dataset)+'.pt'), map_location=device)
    
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

    ray = np.array([0.01,0.99])
    num_samples = 0
    batch = next(iter(test_loader))
    images, ys = batch
    images = images.view(-1, 1, 36, 36)
    images = images.to(device)
    background = images[0:100]
    test_images = images[250:255]
    hnet.eval()
    hnet.zero_grad()
    ray = torch.from_numpy(ray.astype(np.float32)).to(device)

    weights = hnet(ray)
    net = LeNetTarget1([9, 5],weights=weights).to(device)
    print(ys[250:255])
    # masker = shap.maskers.Image("inpaint_telea", images[0].shape)
    # print(masker)
    # explainer = shap.Explainer(net, background)

    # # here we use 500 evaluations of the underlying model to estimate the SHAP values
    # shap_values = explainer(test_images, max_evals=500, batch_size=50, outputs=shap.Explanation.argsort.flip[:1])
    # shap.image_plot(shap_values)
    e = shap.GradientExplainer(net,background)
    shap_values = e.shap_values(test_images)

    shap_numpy = [np.swapaxes(np.swapaxes(s, 1, -1), 1, 2) for s in shap_values]
    test_numpy = np.swapaxes(np.swapaxes(test_images.numpy(), 1, -1), 1, 2)
    shap.image_plot(shap_numpy, -test_numpy,show=False)
    #plt.savefig('scratch2.png')
    shap.image_plot(shap_numpy, -test_numpy,show=False)
    plt.legend(fontsize=18)
    #plt.tight_layout()
    plt.savefig('right.jpg')
    #plt.savefig('left.pdf')

dataset = 'mnist'
path_data = '/home/tuantran/pareto-hypernetworks/data/ParetoMTL_multiMNIST'
save_weights = '/home/tuantran/pareto-hypernetworks/outputs'
#device =  torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
device =  torch.device("cpu")
mode = 'acc'
visualize(dataset,path_data,save_weights,device,mode)

