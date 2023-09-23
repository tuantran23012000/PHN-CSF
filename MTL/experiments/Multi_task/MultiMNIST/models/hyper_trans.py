from typing import List

import torch
import torch.nn.functional as F
from torch import nn
import torchvision.models as models
import math
class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, n_head):
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.attention = ScaleDotProductAttention()
        self.w_q = nn.Linear(d_model, self.n_head)
        self.w_k = nn.Linear(d_model, self.n_head)
        self.w_v = nn.Linear(d_model, self.n_head)
        self.w_concat = nn.Linear(self.n_head, d_model)

    def forward(self, q, k, v, mask=None):
        # 1. dot product with weight matrices
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)

        # 2. split tensor by number of heads
        q, k, v = self.split(q), self.split(k), self.split(v)

        # 3. do scale dot product to compute similarity
        out, attention = self.attention(q, k, v, mask=mask)
        
        # 4. concat and pass to linear layer
        out = self.concat(out)
        out = self.w_concat(out)

        # 5. visualize attention map
        # TODO : we should implement visualization

        return out

    def split(self, tensor):
        """
        split tensor by number of head

        :param tensor: [batch_size, length, d_model]
        :return: [batch_size, head, length, d_tensor]
        """
        batch_size, length, d_model = tensor.size()

        d_tensor = d_model // self.n_head
        tensor = tensor.view(batch_size, length, self.n_head, d_tensor).transpose(1, 2)
        # it is similar with group convolution (split by number of heads)

        return tensor

    def concat(self, tensor):
        """
        inverse function of self.split(tensor : torch.Tensor)

        :param tensor: [batch_size, head, length, d_tensor]
        :return: [batch_size, length, d_model]
        """
        batch_size, head, length, d_tensor = tensor.size()
        d_model = head * d_tensor

        tensor = tensor.transpose(1, 2).contiguous().view(batch_size, length, d_model)
        return tensor
class ScaleDotProductAttention(nn.Module):
    """
    compute scale dot product attention

    Query : given sentence that we focused on (decoder)
    Key : every sentence to check relationship with Qeury(encoder)
    Value : every sentence same with Key (encoder)
    """

    def __init__(self):
        super(ScaleDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None, e=1e-12):
        # input is 4 dimension tensor
        # [batch_size, head, length, d_tensor]
        batch_size, head, length, d_tensor = k.size()

        # 1. dot product Query with Key^T to compute similarity
        k_t = k.transpose(2, 3)  # transpose
        score = (q @ k_t) / math.sqrt(d_tensor)  # scaled dot product

        # 2. apply masking (opt)
        if mask is not None:
            score = score.masked_fill(mask == 0, -10000)

        # 3. pass them softmax to make [0, 1] range
        score = self.softmax(score)

        # 4. multiply with Value
        v = score @ v

        return v, score
class LeNetHyper_trans(nn.Module):
    """LeNet Hypernetwork"""

    def __init__(
        self,
        kernel_size: List[int],
        ray_hidden_dim=100,
        out_dim=10,
        target_hidden_dim=50,
        n_kernels=10,
        n_conv_layers=2,
        n_hidden=1,
        n_tasks=2,
    ):
        super().__init__()
        self.n_conv_layers = n_conv_layers
        self.n_hidden = n_hidden
        self.n_tasks = n_tasks

        assert len(kernel_size) == n_conv_layers, (
            "kernel_size is list with same dim as number of "
            "conv layers holding kernel size for each conv layer"
        )

        # self.ray_mlp = nn.Sequential(
        #     nn.Linear(2, ray_hidden_dim),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(ray_hidden_dim, ray_hidden_dim),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(ray_hidden_dim, ray_hidden_dim),
        # )
        #self.output_layer =  nn.Linear(ray_hidden_dim, n_kernels * kernel_size[0] * kernel_size[0])
        if self.n_tasks == 2:
            self.embedding_layer1 =  nn.Sequential(nn.Linear(1, ray_hidden_dim),nn.ReLU(inplace=True))
            self.embedding_layer2 =  nn.Sequential(nn.Linear(1, ray_hidden_dim),nn.ReLU(inplace=True))
        else:
            self.embedding_layer1 =  nn.Sequential(nn.Linear(1, ray_hidden_dim),nn.ReLU(inplace=True))
            self.embedding_layer2 =  nn.Sequential(nn.Linear(1, ray_hidden_dim),nn.ReLU(inplace=True))
            self.embedding_layer3 =  nn.Sequential(nn.Linear(1, ray_hidden_dim),nn.ReLU(inplace=True))
        self.attention = nn.MultiheadAttention(embed_dim=ray_hidden_dim, num_heads=1) #MultiHeadAttention(d_model=ray_hidden_dim, n_head=2) #nn.MultiheadAttention(embed_dim=ray_hidden_dim, num_heads=2)
        #self.pos_embedding = nn.Parameter(torch.randn(self.n_tasks, ray_hidden_dim))
        self.ffn1 = nn.Linear(ray_hidden_dim,ray_hidden_dim)
        out_hd_dim = ray_hidden_dim
        self.ffn2 = nn.Linear(ray_hidden_dim,ray_hidden_dim)
        
        #self.ffn3 = nn.Linear(self.n_tasks,1)
        self.output_layer = nn.Linear(ray_hidden_dim, out_hd_dim) #nn.Sequential(nn.Linear(ray_hidden_dim,ray_hidden_dim), nn.ReLU(inplace=True))  #nn.Linear(ray_hidden_dim, ray_hidden_dim)
        self.conv_0_weights = nn.Linear(
            out_hd_dim, n_kernels * kernel_size[0] * kernel_size[0]
        )
        self.conv_0_bias = nn.Linear(out_hd_dim, n_kernels)

        for i in range(1, n_conv_layers):
            # previous number of kernels
            p = 2 ** (i - 1) * n_kernels
            # current number of kernels
            c = 2 ** i * n_kernels

            setattr(
                self,
                f"conv_{i}_weights",
                nn.Linear(out_hd_dim, c * p * kernel_size[i] * kernel_size[i]),
            )
            setattr(self, f"conv_{i}_bias", nn.Linear(out_hd_dim, c))

        latent = 25
        self.hidden_0_weights = nn.Linear(
            out_hd_dim, target_hidden_dim * 2 ** i * n_kernels * latent
        )
        self.hidden_0_bias = nn.Linear(out_hd_dim, target_hidden_dim)

        for j in range(n_tasks):
            setattr(
                self,
                f"task_{j}_weights",
                nn.Linear(out_hd_dim, target_hidden_dim * out_dim),
            )
            setattr(self, f"task_{j}_bias", nn.Linear(out_hd_dim, out_dim))

    def shared_parameters(self):
        return list([p for n, p in self.named_parameters() if "task" not in n])

    def forward(self, ray):
        if ray.shape[0] == 2:
                x = torch.stack((self.embedding_layer1(ray[0].unsqueeze(0)),self.embedding_layer2(ray[1].unsqueeze(0))))
        else:
            x = torch.stack((self.embedding_layer1(ray[0].unsqueeze(0)),self.embedding_layer2(ray[1].unsqueeze(0)),self.embedding_layer3(ray[2].unsqueeze(0))))
        #x += self.pos_embedding[:,:]
        x_ = x
        x = x.unsqueeze(1)
        x,_ = self.attention(x,x,x)
        x = x.squeeze(1)
        x = x + x_
        x_ = x
        x = self.ffn1(x)
        x = F.relu(x)
        x = self.ffn2(x)
        x = x + x_
        x = self.output_layer(x)
        x = F.relu(x) 
        features = torch.mean(x,dim=0)

        out_dict = {}
        layer_types = ["conv", "hidden", "task"]

        for i in layer_types:
            if i == "conv":
                n_layers = self.n_conv_layers
            elif i == "hidden":
                n_layers = self.n_hidden
            elif i == "task":
                n_layers = self.n_tasks

            for j in range(n_layers):
                out_dict[f"{i}{j}.weights"] = getattr(self, f"{i}_{j}_weights")(
                    features
                )
                out_dict[f"{i}{j}.bias"] = getattr(self, f"{i}_{j}_bias")(
                    features
                ).flatten()

        return out_dict


class LeNetTarget_trans(nn.Module):
    """LeNet target network"""

    def __init__(
        self,
        kernel_size,
        n_kernels=10,
        out_dim=10,
        target_hidden_dim=50,
        n_conv_layers=2,
        n_tasks=2,
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

    def forward(self, x, weights=None):
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
            logits.append(
                F.linear(
                    x,
                    weight=weights[f"task{j}.weights"].reshape(
                        self.out_dim, self.target_hidden_dim
                    ),
                    bias=weights[f"task{j}.bias"],
                )
            )
        return logits
    def compute_l1_loss(self, w):
        return torch.square(w).sum()
