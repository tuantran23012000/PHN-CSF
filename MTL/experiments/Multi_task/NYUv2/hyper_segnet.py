from typing import List

import torch
import torch.nn.functional as F
from torch import nn
import torchvision.models as models

class SegNetHyper(nn.Module):
    def __init__(
        self,
        preference_dim=3,
        preference_embedding_dim=32,
        hidden_dim=100,
        num_chunks=105,
        chunk_embedding_dim=64,
        num_ws=11,
        w_dim=10000,
        out_task1 = 13,
        out_task2 = 1,
        out_task3 = 3,
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
        self.out_task1 = out_task1
        self.out_task2 = out_task2
        self.out_task3 = out_task3
        # initialization
        torch.nn.init.normal_(
            self.preference_embedding_matrix.weight, mean=0.0, std=0.1
        )
        torch.nn.init.normal_(self.chunk_embedding_matrix.weight, mean=0.0, std=0.1)
        for w in self.ws:
            torch.nn.init.normal_(w, mean=0.0, std=0.1)

        self.layer_to_shape = {
            "segnet.encoder_block.0.conv2d.weight": torch.Size([64, 3, 3, 3]),
            "segnet.encoder_block.0.bn1.weight": torch.Size([64]),
            "segnet.encoder_block.0.bn1.bias": torch.Size([64]),
            "segnet.encoder_block.1.conv2d.weight": torch.Size([128, 64, 3, 3]),
            "segnet.encoder_block.1.bn1.weight": torch.Size([128]),
            "segnet.encoder_block.1.bn1.bias": torch.Size([128]),
            "segnet.encoder_block.2.conv2d.weight": torch.Size([256, 128, 3, 3]),
            "segnet.encoder_block.2.bn1.weight": torch.Size([256]),
            "segnet.encoder_block.2.bn1.bias": torch.Size([256]),
            "segnet.encoder_block.3.conv2d.weight": torch.Size([512, 256, 3, 3]),
            "segnet.encoder_block.3.bn1.weight": torch.Size([512]),
            "segnet.encoder_block.3.bn1.bias": torch.Size([512]),
            "segnet.encoder_block.4.conv2d.weight": torch.Size([512, 512, 3, 3]),
            "segnet.encoder_block.4.bn1.weight": torch.Size([512]),
            "segnet.encoder_block.4.bn1.bias": torch.Size([512]),
            "segnet.decoder_block.0.conv2d.weight": torch.Size([64, 64, 3, 3]),
            "segnet.decoder_block.0.bn1.weight": torch.Size([64]),
            "segnet.decoder_block.0.bn1.bias": torch.Size([64]),
            "segnet.decoder_block.1.conv2d.weight": torch.Size([64, 128, 3, 3]),
            "segnet.decoder_block.1.bn1.weight": torch.Size([64]),
            "segnet.decoder_block.1.bn1.bias": torch.Size([64]),
            "segnet.decoder_block.2.conv2d.weight": torch.Size([128,256, 3, 3]),
            "segnet.decoder_block.2.bn1.weight": torch.Size([128]),
            "segnet.decoder_block.2.bn1.bias": torch.Size([128]),
            "segnet.decoder_block.3.conv2d.weight": torch.Size([256, 512, 3, 3]),
            "segnet.decoder_block.3.bn1.weight": torch.Size([256]),
            "segnet.decoder_block.3.bn1.bias": torch.Size([256]),
            "segnet.decoder_block.4.conv2d.weight": torch.Size([512, 512, 3, 3]),
            "segnet.decoder_block.4.bn1.weight": torch.Size([512]),
            "segnet.decoder_block.4.bn1.bias": torch.Size([512]),
            "segnet.conv_block_enc.0.conv2d.weight": torch.Size([64, 64, 3, 3]),
            "segnet.conv_block_enc.0.bn1.weight": torch.Size([64]),
            "segnet.conv_block_enc.0.bn1.bias": torch.Size([64]),
            "segnet.conv_block_enc.1.conv2d.weight": torch.Size([128, 128, 3, 3]),
            "segnet.conv_block_enc.1.bn1.weight": torch.Size([128]),
            "segnet.conv_block_enc.1.bn1.bias": torch.Size([128]),
            "segnet.conv_block_enc.2.conv2d1.weight": torch.Size([256, 256, 3, 3]),
            "segnet.conv_block_enc.2.bn1.weight": torch.Size([256]),
            "segnet.conv_block_enc.2.bn1.bias": torch.Size([256]),
            "segnet.conv_block_enc.2.conv2d2.weight": torch.Size([256, 256, 3, 3]),
            "segnet.conv_block_enc.2.bn2.weight": torch.Size([256]),
            "segnet.conv_block_enc.2.bn2.bias": torch.Size([256]),
            "segnet.conv_block_enc.3.conv2d1.weight": torch.Size([512, 512, 3, 3]),
            "segnet.conv_block_enc.3.bn1.weight": torch.Size([512]),
            "segnet.conv_block_enc.3.bn1.bias": torch.Size([512]),
            "segnet.conv_block_enc.3.conv2d2.weight": torch.Size([512, 512, 3, 3]),
            "segnet.conv_block_enc.3.bn2.weight": torch.Size([512]),
            "segnet.conv_block_enc.3.bn2.bias": torch.Size([512]),
            "segnet.conv_block_enc.4.conv2d1.weight": torch.Size([512, 512, 3, 3]),
            "segnet.conv_block_enc.4.bn1.weight": torch.Size([512]),
            "segnet.conv_block_enc.4.bn1.bias": torch.Size([512]),
            "segnet.conv_block_enc.4.conv2d2.weight": torch.Size([512, 512, 3, 3]),
            "segnet.conv_block_enc.4.bn2.weight": torch.Size([512]),
            "segnet.conv_block_enc.4.bn2.bias": torch.Size([512]),
            "segnet.conv_block_dec.0.conv2d.weight": torch.Size([64, 64, 3, 3]),
            "segnet.conv_block_dec.0.bn1.weight": torch.Size([64]),
            "segnet.conv_block_dec.0.bn1.bias": torch.Size([64]),
            "segnet.conv_block_dec.1.conv2d.weight": torch.Size([64, 64, 3, 3]),
            "segnet.conv_block_dec.1.bn1.weight": torch.Size([64]),
            "segnet.conv_block_dec.1.bn1.bias": torch.Size([64]),
            "segnet.conv_block_dec.2.conv2d1.weight": torch.Size([128, 128, 3, 3]),
            "segnet.conv_block_dec.2.bn1.weight": torch.Size([128]),
            "segnet.conv_block_dec.2.bn1.bias": torch.Size([128]),
            "segnet.conv_block_dec.2.conv2d2.weight": torch.Size([128, 128, 3, 3]),
            "segnet.conv_block_dec.2.bn2.weight": torch.Size([128]),
            "segnet.conv_block_dec.2.bn2.bias": torch.Size([128]),
            "segnet.conv_block_dec.3.conv2d1.weight": torch.Size([256, 256, 3, 3]),
            "segnet.conv_block_dec.3.bn1.weight": torch.Size([256]),
            "segnet.conv_block_dec.3.bn1.bias": torch.Size([256]),
            "segnet.conv_block_dec.3.conv2d2.weight": torch.Size([256, 256, 3, 3]),
            "segnet.conv_block_dec.3.bn2.weight": torch.Size([256]),
            "segnet.conv_block_dec.3.bn2.bias": torch.Size([256]),
            "segnet.conv_block_dec.4.conv2d1.weight": torch.Size([512, 512, 3, 3]),
            "segnet.conv_block_dec.4.bn1.weight": torch.Size([512]),
            "segnet.conv_block_dec.4.bn1.bias": torch.Size([512]),
            "segnet.conv_block_dec.4.conv2d2.weight": torch.Size([512, 512, 3, 3]),
            "segnet.conv_block_dec.4.bn2.weight": torch.Size([512]),
            "segnet.conv_block_dec.4.bn2.bias": torch.Size([512]),
            "segnet.pred_task1.conv2d1.weight": torch.Size([64, 64, 3, 3]),
            "segnet.pred_task1.conv2d2.weight": torch.Size([13, 64, 1, 1]),
            "segnet.pred_task2.conv2d1.weight": torch.Size([64, 64, 3, 3]),
            "segnet.pred_task2.conv2d2.weight": torch.Size([1, 64, 1, 1]),
            "segnet.pred_task3.conv2d1.weight": torch.Size([64, 64, 3, 3]),
            "segnet.pred_task3.conv2d2.weight": torch.Size([3, 64, 1, 1]),
        }
        
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


class SegNetTarget(nn.Module):
    def __init__(self, pretrained=False, progress=True,momentum=None,model_type="standard", n_tasks = None,**kwargs):
        super().__init__()
        self.momentum = momentum
        self.n_tasks = n_tasks
        self.model_type = model_type
    def forward(self, x, weights=None):
        g_encoder, g_decoder, g_maxpool, g_upsampl, indices = (
            [0] * 5 for _ in range(5)
        )
        for i in range(5):
            g_encoder[i], g_decoder[-i - 1] = ([0] * 2 for _ in range(2))
        # global shared encoder-decoder network
        for i in range(5):
            if i == 0:
                g_encoder[i][0] = self.encoder_block(x,weights,i)
                g_encoder[i][1] = self.conv_block_enc(g_encoder[i][0],weights,i)
                g_maxpool[i], indices[i] = F.max_pool2d(g_encoder[i][1], kernel_size=2, stride=2,return_indices=True) 
                
            else:
                g_encoder[i][0] = self.encoder_block(g_maxpool[i - 1],weights,i)
                g_encoder[i][1] = self.conv_block_enc(g_encoder[i][0],weights,i)
                g_maxpool[i], indices[i] = F.max_pool2d(g_encoder[i][1], kernel_size=2, stride=2,return_indices=True) 
        for i in range(5):
            if i == 0:
                g_upsampl[i] = F.max_unpool2d(g_maxpool[-1], indices[-i - 1],kernel_size=2, stride=2)
                g_decoder[i][0] = self.decoder_block(g_upsampl[i],weights,4-i)
                g_decoder[i][1] = self.conv_block_dec(g_decoder[i][0],weights,4-i)
            else:
                g_upsampl[i] = F.max_unpool2d(g_decoder[i - 1][-1], indices[-i-1],kernel_size=2, stride=2)
                g_decoder[i][0] = self.decoder_block(g_upsampl[i],weights,4-i)
                g_decoder[i][1] = self.conv_block_dec(g_decoder[i][0],weights,4-i)
        t1_pred = F.log_softmax(self.forward_clf(g_decoder[i][1],weights,1), dim=1)
        t2_pred = self.forward_clf(g_decoder[i][1],weights,2)
        # t3_pred = self.forward_clf(g_decoder[i][1],weights,3)
        # t3_pred = t3_pred / torch.norm(t3_pred, p=2, dim=1, keepdim=True)

        return [t1_pred, t2_pred]

    def encoder_block(self,x, weights,index):
        """Before blocks"""
        device = x.device
        x = F.conv2d(x, weights[f"segnet.encoder_block.{index}.conv2d.weight"], stride=1, padding=1)

        x = F.batch_norm(
            x,
            torch.zeros(x.data.size()[1]).to(device),
            torch.ones(x.data.size()[1]).to(device),
            weights[f"segnet.encoder_block.{index}.bn1.weight"],
            weights[f"segnet.encoder_block.{index}.bn1.bias"],
            training=True,
        )
    
        x = F.relu(x)
        return x

    def decoder_block(self,x, weights,index):
        """Before blocks"""
        device = x.device
        
        x = F.conv2d(x, weights[f"segnet.decoder_block.{index}.conv2d.weight"], stride=1, padding=1)

        x = F.batch_norm(
            x,
            torch.zeros(x.data.size()[1]).to(device),
            torch.ones(x.data.size()[1]).to(device),
            weights[f"segnet.decoder_block.{index}.bn1.weight"],
            weights[f"segnet.decoder_block.{index}.bn1.bias"],
            training=True,
        )
    
        x = F.relu(x)
        return x
    def conv_block_enc(self,x, weights,index):
        """Before blocks"""
        device = x.device
        
        if index == 3 or index == 4:
            # conv1
            x = F.conv2d(x, weights[f"segnet.conv_block_enc.{index}.conv2d1.weight"], stride=1, padding=1)
            # bn1
            x = F.batch_norm(
                x,
                torch.zeros(x.data.size()[1]).to(device),
                torch.ones(x.data.size()[1]).to(device),
                weights[f"segnet.conv_block_enc.{index}.bn1.weight"],
                weights[f"segnet.conv_block_enc.{index}.bn1.bias"],
                training=True,
            )
            x = F.relu(x)
            # conv2
            x = F.conv2d(x, weights[f"segnet.conv_block_enc.{index}.conv2d2.weight"], stride=1, padding=1)
            # bn2
            x = F.batch_norm(
                x,
                torch.zeros(x.data.size()[1]).to(device),
                torch.ones(x.data.size()[1]).to(device),
                weights[f"segnet.conv_block_enc.{index}.bn2.weight"],
                weights[f"segnet.conv_block_enc.{index}.bn2.bias"],
                training=True,
            )
            x = F.relu(x)
        return x
    def conv_block_dec(self,x, weights,index):
        """Before blocks"""
        device = x.device
        
        if index == 3 or index == 4:
            # conv1
            x = F.conv2d(x, weights[f"segnet.conv_block_dec.{index}.conv2d1.weight"], stride=1, padding=1)
            # bn1
            x = F.batch_norm(
                x,
                torch.zeros(x.data.size()[1]).to(device),
                torch.ones(x.data.size()[1]).to(device),
                weights[f"segnet.conv_block_dec.{index}.bn1.weight"],
                weights[f"segnet.conv_block_dec.{index}.bn1.bias"],
                training=True,
            )
            x = F.relu(x)
            # conv2
            x = F.conv2d(x, weights[f"segnet.conv_block_dec.{index}.conv2d2.weight"], stride=1, padding=1)
            # bn2
            x = F.batch_norm(
                x,
                torch.zeros(x.data.size()[1]).to(device),
                torch.ones(x.data.size()[1]).to(device),
                weights[f"segnet.conv_block_dec.{index}.bn2.weight"],
                weights[f"segnet.conv_block_dec.{index}.bn2.bias"],
                training=True,
            )
            x = F.relu(x)
        return x
    def forward_clf(self,x, weights, index):
        x = F.conv2d(x, weights[f"segnet.pred_task{index}.conv2d1.weight"], stride=1, padding=1)
        x = F.conv2d(x, weights[f"segnet.pred_task{index}.conv2d2.weight"], stride=1, padding=0)
        return x