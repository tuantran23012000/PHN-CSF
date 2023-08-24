import torch
import torch.nn.functional as F
from torch import nn
import torchvision.models as models

class ResnetHyper(nn.Module):
    def __init__(
        self,
        preference_dim=40,
        preference_embedding_dim=32,
        hidden_dim=100,
        num_chunks=105,
        chunk_embedding_dim=64,
        num_ws=11,
        w_dim=10000,
        out_dim = 10,
        model_name = "resnet18",
    ):
        """

        :param preference_dim: preference vector dimension
        :param preference_embedding_dim: preference embedding dimension
        :param hidden_dim: hidden dimension
        :param num_chunks: number of chunks
        :param chunk_embedding_dim: chunks embedding dimension
        :param num_ws: number of W matrices (see paper for details)
        :param w_dim: row dimension of the W matrices
        """
        super().__init__()
        self.model_name = model_name
        if self.model_name == 'resnet18':
            self.last_dim = 512
            self.layer_block = [2,2,2,2]
        elif self.model_name == 'resnet34':
            self.last_dim = 512
            self.layer_block = [3, 4, 6, 3]
        elif self.model_name == 'resnet50':
            self.last_dim = 2048
            self.layer_block = [3, 4, 6, 3]
        elif self.model_name == 'resnet101':
            self.last_dim = 2048
            self.layer_block = [3, 4, 23, 3]
        self.preference_embedding_dim = preference_embedding_dim
        self.num_chunks = num_chunks
        self.chunk_embedding_matrix = nn.Embedding(
            num_embeddings=num_chunks, embedding_dim=chunk_embedding_dim
        )
        self.preference_embedding_matrix = nn.Embedding(
            num_embeddings=preference_dim, embedding_dim=preference_embedding_dim
        )

        self.fc = nn.Sequential(
            nn.Linear(preference_embedding_dim + chunk_embedding_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )
        list_ws = [self._init_w((w_dim, hidden_dim)) for _ in range(num_ws)]
        self.ws = nn.ParameterList(list_ws)
        self.out_dim = out_dim
        # initialization
        torch.nn.init.normal_(
            self.preference_embedding_matrix.weight, mean=0.0, std=0.1
        )
        torch.nn.init.normal_(self.chunk_embedding_matrix.weight, mean=0.0, std=0.1)
        for w in self.ws:
            torch.nn.init.normal_(w, mean=0.0, std=0.1)

        self.layer1 = {}
        for i in range(self.layer_block[0]):
            if self.model_name == "resnet18" or self.model_name == "resnet34":
                if i == 0:
                    block_update = {
                        "resnet.layer1.0.conv1.weight": torch.Size([64, 64, 3, 3]),
                        "resnet.layer1.0.bn1.weight": torch.Size([64]),
                        "resnet.layer1.0.bn1.bias": torch.Size([64]),
                        "resnet.layer1.0.conv2.weight": torch.Size([64, 64, 3, 3]),
                        "resnet.layer1.0.bn2.weight": torch.Size([64]),
                        "resnet.layer1.0.bn2.bias": torch.Size([64]),
                        }
                else:
                    block_update = {
                        f"resnet.layer1.{i}.conv1.weight": torch.Size([64, 64, 3, 3]),
                        f"resnet.layer1.{i}.bn1.weight": torch.Size([64]),
                        f"resnet.layer1.{i}.bn1.bias": torch.Size([64]),
                        f"resnet.layer1.{i}.conv2.weight": torch.Size([64, 64, 3, 3]),
                        f"resnet.layer1.{i}.bn2.weight": torch.Size([64]),
                        f"resnet.layer1.{i}.bn2.bias": torch.Size([64]),
                        }
            elif self.model_name == "resnet50" or self.model_name == "resnet101":
                if i == 0:
                    block_update = {
                        "resnet.layer1.0.conv1.weight": torch.Size([64, 64, 1, 1]),
                        "resnet.layer1.0.bn1.weight": torch.Size([64]),
                        "resnet.layer1.0.bn1.bias": torch.Size([64]),
                        "resnet.layer1.0.conv2.weight": torch.Size([64, 64, 3, 3]),
                        "resnet.layer1.0.bn2.weight": torch.Size([64]),
                        "resnet.layer1.0.bn2.bias": torch.Size([64]),
                        "resnet.layer1.0.conv3.weight": torch.Size([256, 64, 1, 1]),
                        "resnet.layer1.0.bn3.weight": torch.Size([256]),
                        "resnet.layer1.0.bn3.bias": torch.Size([256]),
                        "resnet.layer1.0.downsample.0.weight": torch.Size([256, 64, 1, 1]),
                        "resnet.layer1.0.downsample.1.weight": torch.Size([256]),
                        "resnet.layer1.0.downsample.1.bias": torch.Size([256]),
                        }
                else:
                    block_update = {
                        f"resnet.layer1.{i}.conv1.weight": torch.Size([64, 256, 1, 1]),
                        f"resnet.layer1.{i}.bn1.weight": torch.Size([64]),
                        f"resnet.layer1.{i}.bn1.bias": torch.Size([64]),
                        f"resnet.layer1.{i}.conv2.weight": torch.Size([64, 64, 3, 3]),
                        f"resnet.layer1.{i}.bn2.weight": torch.Size([64]),
                        f"resnet.layer1.{i}.bn2.bias": torch.Size([64]),
                        f"resnet.layer1.{i}.conv3.weight": torch.Size([256, 64, 1, 1]),
                        f"resnet.layer1.{i}.bn3.weight": torch.Size([256]),
                        f"resnet.layer1.{i}.bn3.bias": torch.Size([256]),
                        }
            self.layer1.update(block_update)
                
        self.layer2 = {}
        for i in range(self.layer_block[1]):
            if self.model_name == "resnet18" or self.model_name == "resnet34":
                if i == 0:
                    block_update = {
                        "resnet.layer2.0.conv1.weight": torch.Size([128, 64, 3, 3]),
                        "resnet.layer2.0.bn1.weight": torch.Size([128]),
                        "resnet.layer2.0.bn1.bias": torch.Size([128]),
                        "resnet.layer2.0.conv2.weight": torch.Size([128, 128, 3, 3]),
                        "resnet.layer2.0.bn2.weight": torch.Size([128]),
                        "resnet.layer2.0.bn2.bias": torch.Size([128]),
                        "resnet.layer2.0.downsample.0.weight": torch.Size([128, 64, 1, 1]),
                        "resnet.layer2.0.downsample.1.weight": torch.Size([128]),
                        "resnet.layer2.0.downsample.1.bias": torch.Size([128]),
                        }
                else:
                    block_update = {
                        f"resnet.layer2.{i}.conv1.weight": torch.Size([128, 128, 3, 3]),
                        f"resnet.layer2.{i}.bn1.weight": torch.Size([128]),
                        f"resnet.layer2.{i}.bn1.bias": torch.Size([128]),
                        f"resnet.layer2.{i}.conv2.weight": torch.Size([128, 128, 3, 3]),
                        f"resnet.layer2.{i}.bn2.weight": torch.Size([128]),
                        f"resnet.layer2.{i}.bn2.bias": torch.Size([128]),
                        }
            elif self.model_name == "resnet50" or self.model_name == "resnet101":
                if i == 0:
                    block_update = {
                        "resnet.layer2.0.conv1.weight": torch.Size([128, 256, 1, 1]),
                        "resnet.layer2.0.bn1.weight": torch.Size([128]),
                        "resnet.layer2.0.bn1.bias": torch.Size([128]),
                        "resnet.layer2.0.conv2.weight": torch.Size([128, 128, 3, 3]),
                        "resnet.layer2.0.bn2.weight": torch.Size([128]),
                        "resnet.layer2.0.bn2.bias": torch.Size([128]),
                        "resnet.layer2.0.conv3.weight": torch.Size([512, 128, 1, 1]),
                        "resnet.layer2.0.bn3.weight": torch.Size([512]),
                        "resnet.layer2.0.bn3.bias": torch.Size([512]),
                        "resnet.layer2.0.downsample.0.weight": torch.Size([512, 256, 1, 1]),
                        "resnet.layer2.0.downsample.1.weight": torch.Size([512]),
                        "resnet.layer2.0.downsample.1.bias": torch.Size([512]),
                        }
                else:
                    block_update = {
                        f"resnet.layer2.{i}.conv1.weight": torch.Size([128, 512, 1, 1]),
                        f"resnet.layer2.{i}.bn1.weight": torch.Size([128]),
                        f"resnet.layer2.{i}.bn1.bias": torch.Size([128]),
                        f"resnet.layer2.{i}.conv2.weight": torch.Size([128, 128, 3, 3]),
                        f"resnet.layer2.{i}.bn2.weight": torch.Size([128]),
                        f"resnet.layer2.{i}.bn2.bias": torch.Size([128]),
                        f"resnet.layer2.{i}.conv3.weight": torch.Size([512, 128, 1, 1]),
                        f"resnet.layer2.{i}.bn3.weight": torch.Size([512]),
                        f"resnet.layer2.{i}.bn3.bias": torch.Size([512]),
                        }
            self.layer2.update(block_update)

        self.layer3 = {}
        for i in range(self.layer_block[2]):
            if self.model_name == "resnet18" or self.model_name == "resnet34":
                if i == 0:
                    block_update = {
                        "resnet.layer3.0.conv1.weight": torch.Size([256, 128, 3, 3]),
                        "resnet.layer3.0.bn1.weight": torch.Size([256]),
                        "resnet.layer3.0.bn1.bias": torch.Size([256]),
                        "resnet.layer3.0.conv2.weight": torch.Size([256, 256, 3, 3]),
                        "resnet.layer3.0.bn2.weight": torch.Size([256]),
                        "resnet.layer3.0.bn2.bias": torch.Size([256]),
                        "resnet.layer3.0.downsample.0.weight": torch.Size([256, 128, 1, 1]),
                        "resnet.layer3.0.downsample.1.weight": torch.Size([256]),
                        "resnet.layer3.0.downsample.1.bias": torch.Size([256]),
                        }
                else:
                    block_update = {
                        f"resnet.layer3.{i}.conv1.weight": torch.Size([256, 256, 3, 3]),
                        f"resnet.layer3.{i}.bn1.weight": torch.Size([256]),
                        f"resnet.layer3.{i}.bn1.bias": torch.Size([256]),
                        f"resnet.layer3.{i}.conv2.weight": torch.Size([256, 256, 3, 3]),
                        f"resnet.layer3.{i}.bn2.weight": torch.Size([256]),
                        f"resnet.layer3.{i}.bn2.bias": torch.Size([256]),
                        }
            elif self.model_name == "resnet50" or self.model_name == "resnet101":
                if i == 0:
                    block_update = {
                        "resnet.layer3.0.conv1.weight": torch.Size([256, 512, 1, 1]),
                        "resnet.layer3.0.bn1.weight": torch.Size([256]),
                        "resnet.layer3.0.bn1.bias": torch.Size([256]),
                        "resnet.layer3.0.conv2.weight": torch.Size([256, 256, 3, 3]),
                        "resnet.layer3.0.bn2.weight": torch.Size([256]),
                        "resnet.layer3.0.bn2.bias": torch.Size([256]),
                        "resnet.layer3.0.conv3.weight": torch.Size([1024, 256, 1, 1]),
                        "resnet.layer3.0.bn3.weight": torch.Size([1024]),
                        "resnet.layer3.0.bn3.bias": torch.Size([1024]),
                        "resnet.layer3.0.downsample.0.weight": torch.Size([1024, 512, 1, 1]),
                        "resnet.layer3.0.downsample.1.weight": torch.Size([1024]),
                        "resnet.layer3.0.downsample.1.bias": torch.Size([1024]), 
                        }
                else:
                    block_update = {
                        f"resnet.layer3.{i}.conv1.weight": torch.Size([256, 1024, 1, 1]),
                        f"resnet.layer3.{i}.bn1.weight": torch.Size([256]),
                        f"resnet.layer3.{i}.bn1.bias": torch.Size([256]),
                        f"resnet.layer3.{i}.conv2.weight": torch.Size([256, 256, 3, 3]),
                        f"resnet.layer3.{i}.bn2.weight": torch.Size([256]),
                        f"resnet.layer3.{i}.bn2.bias": torch.Size([256]),
                        f"resnet.layer3.{i}.conv3.weight": torch.Size([1024, 256, 1, 1]),
                        f"resnet.layer3.{i}.bn3.weight": torch.Size([1024]),
                        f"resnet.layer3.{i}.bn3.bias": torch.Size([1024]),
                        } 
            self.layer3.update(block_update)

        self.layer4 = {}
        for i in range(self.layer_block[3]):
            if self.model_name == "resnet18" or self.model_name == "resnet34":
                if i == 0:
                    block_update = {
                        "resnet.layer4.0.conv1.weight": torch.Size([512, 256, 3, 3]),
                        "resnet.layer4.0.bn1.weight": torch.Size([512]),
                        "resnet.layer4.0.bn1.bias": torch.Size([512]),
                        "resnet.layer4.0.conv2.weight": torch.Size([512, 512, 3, 3]),
                        "resnet.layer4.0.bn2.weight": torch.Size([512]),
                        "resnet.layer4.0.bn2.bias": torch.Size([512]),
                        "resnet.layer4.0.downsample.0.weight": torch.Size([512, 256, 1, 1]),
                        "resnet.layer4.0.downsample.1.weight": torch.Size([512]),
                        "resnet.layer4.0.downsample.1.bias": torch.Size([512]),
                        }
                else:
                    block_update = {
                        f"resnet.layer4.{i}.conv1.weight": torch.Size([512, 512, 3, 3]),
                        f"resnet.layer4.{i}.bn1.weight": torch.Size([512]),
                        f"resnet.layer4.{i}.bn1.bias": torch.Size([512]),
                        f"resnet.layer4.{i}.conv2.weight": torch.Size([512, 512, 3, 3]),
                        f"resnet.layer4.{i}.bn2.weight": torch.Size([512]),
                        f"resnet.layer4.{i}.bn2.bias": torch.Size([512]),
                        }
            elif self.model_name == "resnet50" or self.model_name == "resnet101":
                if i == 0:
                    block_update = {
                        "resnet.layer4.0.conv1.weight": torch.Size([512, 1024, 1,1]),
                        "resnet.layer4.0.bn1.weight": torch.Size([512]),
                        "resnet.layer4.0.bn1.bias": torch.Size([512]),
                        "resnet.layer4.0.conv2.weight": torch.Size([512, 512, 3, 3]),
                        "resnet.layer4.0.bn2.weight": torch.Size([512]),
                        "resnet.layer4.0.bn2.bias": torch.Size([512]),
                        "resnet.layer4.0.conv3.weight": torch.Size([2048, 512, 1, 1]),
                        "resnet.layer4.0.bn3.weight": torch.Size([2048]),
                        "resnet.layer4.0.bn3.bias": torch.Size([2048]),
                        "resnet.layer4.0.downsample.0.weight": torch.Size([2048, 1024, 1, 1]),
                        "resnet.layer4.0.downsample.1.weight": torch.Size([2048]),
                        "resnet.layer4.0.downsample.1.bias": torch.Size([2048]),
                        }
                else:
                    block_update = {
                        f"resnet.layer4.{i}.conv1.weight": torch.Size([512, 2048, 1,1]),
                        f"resnet.layer4.{i}.bn1.weight": torch.Size([512]),
                        f"resnet.layer4.{i}.bn1.bias": torch.Size([512]),
                        f"resnet.layer4.{i}.conv2.weight": torch.Size([512, 512, 3, 3]),
                        f"resnet.layer4.{i}.bn2.weight": torch.Size([512]),
                        f"resnet.layer4.{i}.bn2.bias": torch.Size([512]),
                        f"resnet.layer4.{i}.conv3.weight": torch.Size([2048, 512, 1, 1]),
                        f"resnet.layer4.{i}.bn3.weight": torch.Size([2048]),
                        f"resnet.layer4.{i}.bn3.bias": torch.Size([2048]), 
                        }
            self.layer4.update(block_update)
                
        self.out_layer = {}
        for i in range(preference_dim):
            last_layer = {
                f"task{i+1}.weight": torch.Size([self.out_dim, self.last_dim]),
                f"task{i+1}.bias": torch.Size([self.out_dim]),
                }
            self.out_layer.update(last_layer)
        self.in_layer = {
        "resnet.conv1.weight": torch.Size(
            [64, 3, 7, 7]
        ),  # torch.Size([64, 3, 7, 7]),
        "resnet.bn1.weight": torch.Size([64]),
        "resnet.bn1.bias": torch.Size([64]),
        }
        self.layer_to_shape = {**self.in_layer,**self.layer1,**self.layer2,**self.layer3,**self.layer4,**self.out_layer}

    def _init_w(self, shapes):
        return nn.Parameter(torch.randn(shapes), requires_grad=True)

    def forward(self, preference):
        # preference embedding
        pref_embedding = torch.zeros(
            (self.preference_embedding_dim,), device=preference.device
        )
        for i, pref in enumerate(preference):
            #print(i,pref)
            pref_embedding += (
                self.preference_embedding_matrix(
                    torch.tensor([i], device=preference.device)
                ).squeeze(0)
                * pref
            )
        # chunk embedding
        weights = []
        for chunk_id in range(self.num_chunks):
            chunk_embedding = self.chunk_embedding_matrix(
                torch.tensor([chunk_id], device=preference.device)
            ).squeeze(0)
            # input to fc
            input_embedding = torch.cat((pref_embedding, chunk_embedding)).unsqueeze(0)
            # hidden representation
            rep = self.fc(input_embedding)

            weights.append(torch.cat([F.linear(rep, weight=w) for w in self.ws], dim=1))

        weight_vector = torch.cat(weights, dim=1).squeeze(0)

        out_dict = dict()
        position = 0
        for name, shapes in self.layer_to_shape.items():
            out_dict[name] = weight_vector[
                position : position + shapes.numel()
            ].reshape(shapes)
            position += shapes.numel()
        return out_dict


class ResNetTarget(nn.Module):
    def __init__(self, pretrained=False, progress=True,momentum=None, model_name=None,n_tasks = None,**kwargs):
        super().__init__()
        self.model_name = model_name
        if self.model_name == 'resnet18':
            self.last_dim = 512
            self.layer_block = [2,2,2,2]
        elif self.model_name == 'resnet34':
            self.last_dim = 512
            self.layer_block = [3, 4, 6, 3]
        elif self.model_name == 'resnet50':
            self.last_dim = 2048
            self.layer_block = [3, 4, 6, 3]
        elif self.model_name == 'resnet101':
            self.last_dim = 2048
            self.layer_block = [3, 4, 23, 3]
        self.resnet = models.resnet50(
            pretrained=True, progress=progress, num_classes=1000, **kwargs
        )
        self.resnet.conv1.weight.data = torch.randn((64, 1, 7, 7))

        self.task1 = nn.Linear(512, 10)
        self.task2 = nn.Linear(512, 10)
        self.momentum = momentum
        self.n_tasks = n_tasks
    def forward(self, x, weights=None):
        # pad input
        x = F.pad(input=x, pad=[0, 2, 0, 2], mode="constant", value=0.0)
        if weights is None:
            x = self.resnet(x)
            x = F.relu(x)
            p1, p2 = self.task1(x), self.task2(x)
            return p1, p2

        else:
            x = self.forward_init(x, weights)
            
            for i in range(4):
                x = self.forward_layer(x, weights, i+1)
            x = F.avg_pool2d(x, 4)
            x = x.view(x.size(0), -1)
            out = []
            for i in range(self.n_tasks):
                x1 = self.forward_clf(x, weights, i+1)
                x1 = F.log_softmax(x1, dim=1)
                out.append(x1)
            return out

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
        if self.model_name == "resnet18" or self.model_name == "resnet34":
            # conv1
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
            # conv2
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
        elif self.model_name == "resnet50" or self.model_name == "resnet101":
            # conv1
            out = F.conv2d(
                x,
                weights[f"resnet.layer{layer}.{index}.conv1.weight"],
                stride=1,
                #padding=1,
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
            # conv2
            out = F.conv2d(
                out,
                weights[f"resnet.layer{layer}.{index}.conv2.weight"],
                stride=stride,
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
            out = F.relu(out, inplace=True)

            # conv3
            out = F.conv2d(
                out,
                weights[f"resnet.layer{layer}.{index}.conv3.weight"],
                stride=1,
                #padding=1,
            )
            # bn
            
            out = F.batch_norm(
                out,
                torch.zeros(out.data.size()[1]).to(device),
                torch.ones(out.data.size()[1]).to(device),
                weights[f"resnet.layer{layer}.{index}.bn3.weight"],
                weights[f"resnet.layer{layer}.{index}.bn3.bias"],
                training=True,
            )
            
            if layer > 0 and index == 0:
                identity = self.forward_dowmsample(x, weights, layer)
        out += identity

        out = F.relu(out)

        return out

    @staticmethod
    def forward_dowmsample(x, weights, layer):
        if layer == 1:
            stride = 1
        else:
            stride = 2
        device = x.device

        out = F.conv2d(
            x, weights[f"resnet.layer{layer}.0.downsample.0.weight"], stride=stride
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
        
        for i in range(self.layer_block[layer-1]):
            x = self.forward_block(x, weights, layer,i)
        return x

    @staticmethod
    def forward_linear(x, weights):
        return F.linear(x, weights["resnet.fc.weight"], weights["resnet.fc.bias"])

    @staticmethod
    def forward_clf(x, weights, index):
        return F.linear(x, weights[f"task{index}.weight"], weights[f"task{index}.bias"])