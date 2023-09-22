from torch import nn
import torch.nn.functional as F
import torch
class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-12):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, unbiased=False, keepdim=True)
        # '-1' means last dimension. 

        out = (x - mean) / torch.sqrt(var + self.eps)
        out = self.gamma * out + self.beta
        return out
class Hyper_trans(nn.Module):
      "HyperTransformer"

      def __init__(self, ray_hidden_dim=100, out_dim = 2,n_tasks=2,num_hidden_layer=2,last_activation='relu'):
            super().__init__()
            self.out_dim = out_dim
            self.n_tasks = n_tasks
            self.ray_hidden_dim = ray_hidden_dim
            self.num_hidden_layer = num_hidden_layer
            self.last_activation = last_activation
            if self.n_tasks == 2:
                self.embedding_layer1 =  nn.Sequential(nn.Linear(1, self.ray_hidden_dim),nn.ReLU(inplace=True))
                self.embedding_layer2 =  nn.Sequential(nn.Linear(1, self.ray_hidden_dim),nn.ReLU(inplace=True))
            else:
                self.embedding_layer1 =  nn.Sequential(nn.Linear(1, self.ray_hidden_dim),nn.ReLU(inplace=True))
                self.embedding_layer2 =  nn.Sequential(nn.Linear(1, self.ray_hidden_dim),nn.ReLU(inplace=True))
                self.embedding_layer3 =  nn.Sequential(nn.Linear(1, self.ray_hidden_dim),nn.ReLU(inplace=True))
            self.output_layer =  nn.Linear(self.ray_hidden_dim, self.out_dim)
            self.attention = nn.MultiheadAttention(embed_dim=self.ray_hidden_dim, num_heads=1)
            self.ffn1 = nn.Linear(self.ray_hidden_dim, self.ray_hidden_dim)
            self.ffn2 = nn.Linear(self.ray_hidden_dim, self.ray_hidden_dim)
      def forward(self, ray):
            if ray.shape[0] == 2:
                x = torch.stack((self.embedding_layer1(ray[0].unsqueeze(0)),self.embedding_layer2(ray[1].unsqueeze(0))))
            else:
                x = torch.stack((self.embedding_layer1(ray[0].unsqueeze(0)),self.embedding_layer2(ray[1].unsqueeze(0)),self.embedding_layer3(ray[2].unsqueeze(0))))
            x_ = x
            
            x,_ = self.attention(x,x,x)
            x = x + x_
            x_ = x
            x = self.ffn1(x)
            x = F.relu(x)
            x = self.ffn2(x)
            x = x + x_
            x = self.output_layer(x)
            if self.last_activation == 'relu':
                    x = F.relu(x)
            elif self.last_activation == 'sigmoid':
                    x = F.sigmoid(x)
            elif self.last_activation == 'softmax':
                    x = F.softmax(x)    
            else:
                    x = x
            x = torch.mean(x,dim=0)
            x = x.unsqueeze(0)
            return x
