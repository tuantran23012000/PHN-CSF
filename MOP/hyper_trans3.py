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
class Attention(nn.Module):
    def __init__(self, num_heads, ray_hidden_dim):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.ray_hidden_dim = ray_hidden_dim
        self.multi_head_attention = nn.MultiheadAttention(embed_dim=self.ray_hidden_dim, num_heads=self.num_heads)
        self.ffn1 = nn.Linear(self.ray_hidden_dim, self.ray_hidden_dim)
        self.ffn2 = nn.Linear(self.ray_hidden_dim, self.ray_hidden_dim)
    def forward(self, x):
        x_ = x
        x,_ = self.multi_head_attention(x,x,x)
        x = x + x_
        x_ = x
        x = self.ffn1(x)
        x = F.relu(x)
        x = self.ffn2(x)
        x = x + x_
        return x
class Encode(nn.Module):
    def __init__(self, n_tasks, ray_hidden_dim):
        super(Encode, self).__init__()
        self.n_tasks = n_tasks
        self.ray_hidden_dim = ray_hidden_dim
        if self.n_tasks == 2:
            self.embedding_layer1 =  nn.Sequential(nn.Linear(1, self.ray_hidden_dim),nn.ReLU(inplace=True))
            self.embedding_layer2 =  nn.Sequential(nn.Linear(1, self.ray_hidden_dim),nn.ReLU(inplace=True))
        else:
            self.embedding_layer1 =  nn.Sequential(nn.Linear(1, self.ray_hidden_dim),nn.ReLU(inplace=True))
            self.embedding_layer2 =  nn.Sequential(nn.Linear(1, self.ray_hidden_dim),nn.ReLU(inplace=True))
            self.embedding_layer3 =  nn.Sequential(nn.Linear(1, self.ray_hidden_dim),nn.ReLU(inplace=True))
    def forward(self, ray):
        if ray.shape[0] == 2:
            x = torch.stack((self.embedding_layer1(ray[0].unsqueeze(0)),self.embedding_layer2(ray[1].unsqueeze(0))))
        else:
            x = torch.stack((self.embedding_layer1(ray[0].unsqueeze(0)),self.embedding_layer2(ray[1].unsqueeze(0)),self.embedding_layer3(ray[2].unsqueeze(0))))
        return x
class Decode(nn.Module):
    def __init__(self, out_dim,n_tasks):
        super(Decode, self).__init__()
        self.out_dim = out_dim
        if self.out_dim == 1:
            self.embedding_layer1 =  nn.Linear(n_tasks, 1)
        elif self.out_dim == 2:
            self.embedding_layer1 =  nn.Linear(n_tasks, 1)
            self.embedding_layer2 =  nn.Linear(n_tasks, 1)
        else:
            self.embedding_layer1 =  nn.Linear(n_tasks, 1)
            self.embedding_layer2 =  nn.Linear(n_tasks, 1)
            self.embedding_layer3 =  nn.Linear(n_tasks, 1)
    def forward(self, x):
        if x.shape[0] == 1:
            x = self.embedding_layer1(x[0,:])
            x = x.unsqueeze(0)
        elif x.shape[0] == 2:
            #print("Embed: ",self.embedding_layer1(x[0,:]).shape)
            x = torch.stack((self.embedding_layer1(x[0,:]),self.embedding_layer2(x[1,:])))
        else:
            x = torch.stack((self.embedding_layer1(x[0,:]),self.embedding_layer2(x[1,:]),self.embedding_layer3(x[2,:])))
        return x
class Hyper_trans3(nn.Module):
      "Hypernetwork"

      def __init__(self, ray_hidden_dim=100, out_dim = 2,n_tasks=2,num_hidden_layer=2,last_activation='relu'):
            super().__init__()
            self.out_dim = out_dim
            self.n_tasks = n_tasks
            self.ray_hidden_dim = ray_hidden_dim
            self.num_hidden_layer = num_hidden_layer
            self.last_activation = last_activation
            self.encoder = Encode(n_tasks, ray_hidden_dim)
            self.decoder = Decode(out_dim, n_tasks)
            self.output_layer =  nn.Linear(self.ray_hidden_dim, self.out_dim)
            self.attention = Attention(1,self.ray_hidden_dim)
      def forward(self, ray):
            x = self.encoder(ray)
            x =  self.attention(x)
            x = self.output_layer(x)
            #x = F.relu(x)
            x = x.transpose(1,0)
            x = self.decoder(x)
            x = x.transpose(1,0)
            if self.last_activation == 'relu':
                    x = F.relu(x)
            elif self.last_activation == 'sigmoid':
                    x = F.sigmoid(x)
            elif self.last_activation == 'softmax':
                    x = F.softmax(x)    
            else:
                    x = x
            return x
