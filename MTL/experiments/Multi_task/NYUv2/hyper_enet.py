from typing import List

import torch
import torch.nn.functional as F
from torch import nn
import torchvision.models as models

class ENetHyper(nn.Module):
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
            "segnet.initial_block.conv2d.weight": torch.Size([13, 3, 3, 3]),
            "segnet.initial_block.bn1.weight": torch.Size([16]),
            "segnet.initial_block.bn1.bias": torch.Size([16]),
            "segnet.downsample1_0.ext_conv1.conv2d.weight": torch.Size([4, 16, 2, 2]),
            "segnet.downsample1_0.ext_conv1.bn1.weight": torch.Size([4]),
            "segnet.downsample1_0.ext_conv1.bn1.bias": torch.Size([4]),
            "segnet.downsample1_0.ext_conv2.conv2d.weight": torch.Size([4, 4, 3,3]),
            "segnet.downsample1_0.ext_conv2.bn1.weight": torch.Size([4]),
            "segnet.downsample1_0.ext_conv2.bn1.bias": torch.Size([4]),
            "segnet.downsample1_0.ext_conv3.conv2d.weight": torch.Size([64, 4, 1,1]),
            "segnet.downsample1_0.ext_conv3.bn1.weight": torch.Size([64]),
            "segnet.downsample1_0.ext_conv3.bn1.bias": torch.Size([64]),
            "segnet.regular1_1.ext_conv1.conv2d.weight": torch.Size([16, 64, 1, 1]),
            "segnet.regular1_1.ext_conv1.bn1.weight": torch.Size([16]),
            "segnet.regular1_1.ext_conv1.bn1.bias": torch.Size([16]),
            "segnet.regular1_1.ext_conv2.conv2d.weight": torch.Size([16, 16, 3, 3]),
            "segnet.regular1_1.ext_conv2.bn1.weight": torch.Size([16]),
            "segnet.regular1_1.ext_conv2.bn1.bias": torch.Size([16]),
            "segnet.regular1_1.ext_conv3.conv2d.weight": torch.Size([64, 16, 1, 1]),
            "segnet.regular1_1.ext_conv3.bn1.weight": torch.Size([64]),
            "segnet.regular1_1.ext_conv3.bn1.bias": torch.Size([64]),
            "segnet.regular1_2.ext_conv1.conv2d.weight": torch.Size([16, 64, 1, 1]),
            "segnet.regular1_2.ext_conv1.bn1.weight": torch.Size([16]),
            "segnet.regular1_2.ext_conv1.bn1.bias": torch.Size([16]),
            "segnet.regular1_2.ext_conv2.conv2d.weight": torch.Size([16, 16, 3, 3]),
            "segnet.regular1_2.ext_conv2.bn1.weight": torch.Size([16]),
            "segnet.regular1_2.ext_conv2.bn1.bias": torch.Size([16]),
            "segnet.regular1_2.ext_conv3.conv2d.weight": torch.Size([64, 16, 1, 1]),
            "segnet.regular1_2.ext_conv3.bn1.weight": torch.Size([64]),
            "segnet.regular1_2.ext_conv3.bn1.bias": torch.Size([64]),
            "segnet.regular1_3.ext_conv1.conv2d.weight": torch.Size([16, 64, 1, 1]),
            "segnet.regular1_3.ext_conv1.bn1.weight": torch.Size([16]),
            "segnet.regular1_3.ext_conv1.bn1.bias": torch.Size([16]),
            "segnet.regular1_3.ext_conv2.conv2d.weight": torch.Size([16, 16, 3, 3]),
            "segnet.regular1_3.ext_conv2.bn1.weight": torch.Size([16]),
            "segnet.regular1_3.ext_conv2.bn1.bias": torch.Size([16]),
            "segnet.regular1_3.ext_conv3.conv2d.weight": torch.Size([64, 16, 1, 1]),
            "segnet.regular1_3.ext_conv3.bn1.weight": torch.Size([64]),
            "segnet.regular1_3.ext_conv3.bn1.bias": torch.Size([64]),
            "segnet.regular1_4.ext_conv1.conv2d.weight": torch.Size([16, 64, 1, 1]),
            "segnet.regular1_4.ext_conv1.bn1.weight": torch.Size([16]),
            "segnet.regular1_4.ext_conv1.bn1.bias": torch.Size([16]),
            "segnet.regular1_4.ext_conv2.conv2d.weight": torch.Size([16, 16, 3, 3]),
            "segnet.regular1_4.ext_conv2.bn1.weight": torch.Size([16]),
            "segnet.regular1_4.ext_conv2.bn1.bias": torch.Size([16]),
            "segnet.regular1_4.ext_conv3.conv2d.weight": torch.Size([64, 16, 1, 1]),
            "segnet.regular1_4.ext_conv3.bn1.weight": torch.Size([64]),
            "segnet.regular1_4.ext_conv3.bn1.bias": torch.Size([64]),
            "segnet.downsample2_0.ext_conv1.conv2d.weight": torch.Size([16, 64, 2, 2]),
            "segnet.downsample2_0.ext_conv1.bn1.weight": torch.Size([16]),
            "segnet.downsample2_0.ext_conv1.bn1.bias": torch.Size([16]),
            "segnet.downsample2_0.ext_conv2.conv2d.weight": torch.Size([16, 16, 3,3]),
            "segnet.downsample2_0.ext_conv2.bn1.weight": torch.Size([16]),
            "segnet.downsample2_0.ext_conv2.bn1.bias": torch.Size([16]),
            "segnet.downsample2_0.ext_conv3.conv2d.weight": torch.Size([128, 16, 1,1]),
            "segnet.downsample2_0.ext_conv3.bn1.weight": torch.Size([128]),
            "segnet.downsample2_0.ext_conv3.bn1.bias": torch.Size([128]),
            "segnet.regular2_1.ext_conv1.conv2d.weight": torch.Size([32, 128, 1, 1]),
            "segnet.regular2_1.ext_conv1.bn1.weight": torch.Size([32]),
            "segnet.regular2_1.ext_conv1.bn1.bias": torch.Size([32]),
            "segnet.regular2_1.ext_conv2.conv2d.weight": torch.Size([32, 32, 3, 3]),
            "segnet.regular2_1.ext_conv2.bn1.weight": torch.Size([32]),
            "segnet.regular2_1.ext_conv2.bn1.bias": torch.Size([32]),
            "segnet.regular2_1.ext_conv3.conv2d.weight": torch.Size([128, 32, 1, 1]),
            "segnet.regular2_1.ext_conv3.bn1.weight": torch.Size([128]),
            "segnet.regular2_1.ext_conv3.bn1.bias": torch.Size([128]),
            "segnet.dilated2_2.ext_conv1.conv2d.weight": torch.Size([32, 128, 1, 1]),
            "segnet.dilated2_2.ext_conv1.bn1.weight": torch.Size([32]),
            "segnet.dilated2_2.ext_conv1.bn1.bias": torch.Size([32]),
            "segnet.dilated2_2.ext_conv2.conv2d.weight": torch.Size([32, 32, 3, 3]),
            "segnet.dilated2_2.ext_conv2.bn1.weight": torch.Size([32]),
            "segnet.dilated2_2.ext_conv2.bn1.bias": torch.Size([32]),
            "segnet.dilated2_2.ext_conv3.conv2d.weight": torch.Size([128, 32, 1, 1]),
            "segnet.dilated2_2.ext_conv3.bn1.weight": torch.Size([128]),
            "segnet.dilated2_2.ext_conv3.bn1.bias": torch.Size([128]),
            "segnet.asymmetric2_3.ext_conv1.conv2d.weight": torch.Size([32, 128, 1, 1]),
            "segnet.asymmetric2_3.ext_conv1.bn1.weight": torch.Size([32]),
            "segnet.asymmetric2_3.ext_conv1.bn1.bias": torch.Size([32]),
            "segnet.asymmetric2_3.ext_conv2.conv2d1.weight": torch.Size([32, 32, 5, 1]),
            "segnet.asymmetric2_3.ext_conv2.bn1.weight": torch.Size([32]),
            "segnet.asymmetric2_3.ext_conv2.bn1.bias": torch.Size([32]),
            "segnet.asymmetric2_3.ext_conv2.conv2d2.weight": torch.Size([32, 32, 1, 5]),
            "segnet.asymmetric2_3.ext_conv2.bn2.weight": torch.Size([32]),
            "segnet.asymmetric2_3.ext_conv2.bn2.bias": torch.Size([32]),
            "segnet.asymmetric2_3.ext_conv3.conv2d.weight": torch.Size([128, 32, 1, 1]),
            "segnet.asymmetric2_3.ext_conv3.bn1.weight": torch.Size([128]),
            "segnet.asymmetric2_3.ext_conv3.bn1.bias": torch.Size([128]),
            "segnet.dilated2_4.ext_conv1.conv2d.weight": torch.Size([32, 128, 1, 1]),
            "segnet.dilated2_4.ext_conv1.bn1.weight": torch.Size([32]),
            "segnet.dilated2_4.ext_conv1.bn1.bias": torch.Size([32]),
            "segnet.dilated2_4.ext_conv2.conv2d.weight": torch.Size([32, 32, 3, 3]),
            "segnet.dilated2_4.ext_conv2.bn1.weight": torch.Size([32]),
            "segnet.dilated2_4.ext_conv2.bn1.bias": torch.Size([32]),
            "segnet.dilated2_4.ext_conv3.conv2d.weight": torch.Size([128, 32, 1, 1]),
            "segnet.dilated2_4.ext_conv3.bn1.weight": torch.Size([128]),
            "segnet.dilated2_4.ext_conv3.bn1.bias": torch.Size([128]),
            "segnet.regular2_5.ext_conv1.conv2d.weight": torch.Size([32, 128, 1, 1]),
            "segnet.regular2_5.ext_conv1.bn1.weight": torch.Size([32]),
            "segnet.regular2_5.ext_conv1.bn1.bias": torch.Size([32]),
            "segnet.regular2_5.ext_conv2.conv2d.weight": torch.Size([32, 32, 3, 3]),
            "segnet.regular2_5.ext_conv2.bn1.weight": torch.Size([32]),
            "segnet.regular2_5.ext_conv2.bn1.bias": torch.Size([32]),
            "segnet.regular2_5.ext_conv3.conv2d.weight": torch.Size([128, 32, 1, 1]),
            "segnet.regular2_5.ext_conv3.bn1.weight": torch.Size([128]),
            "segnet.regular2_5.ext_conv3.bn1.bias": torch.Size([128]),
            "segnet.dilated2_6.ext_conv1.conv2d.weight": torch.Size([32, 128, 1, 1]),
            "segnet.dilated2_6.ext_conv1.bn1.weight": torch.Size([32]),
            "segnet.dilated2_6.ext_conv1.bn1.bias": torch.Size([32]),
            "segnet.dilated2_6.ext_conv2.conv2d.weight": torch.Size([32, 32, 3, 3]),
            "segnet.dilated2_6.ext_conv2.bn1.weight": torch.Size([32]),
            "segnet.dilated2_6.ext_conv2.bn1.bias": torch.Size([32]),
            "segnet.dilated2_6.ext_conv3.conv2d.weight": torch.Size([128, 32, 1, 1]),
            "segnet.dilated2_6.ext_conv3.bn1.weight": torch.Size([128]),
            "segnet.dilated2_6.ext_conv3.bn1.bias": torch.Size([128]),
            "segnet.asymmetric2_7.ext_conv1.conv2d.weight": torch.Size([32, 128, 1, 1]),
            "segnet.asymmetric2_7.ext_conv1.bn1.weight": torch.Size([32]),
            "segnet.asymmetric2_7.ext_conv1.bn1.bias": torch.Size([32]),
            "segnet.asymmetric2_7.ext_conv2.conv2d1.weight": torch.Size([32, 32, 5, 1]),
            "segnet.asymmetric2_7.ext_conv2.bn1.weight": torch.Size([32]),
            "segnet.asymmetric2_7.ext_conv2.bn1.bias": torch.Size([32]),
            "segnet.asymmetric2_7.ext_conv2.conv2d2.weight": torch.Size([32, 32, 1, 5]),
            "segnet.asymmetric2_7.ext_conv2.bn2.weight": torch.Size([32]),
            "segnet.asymmetric2_7.ext_conv2.bn2.bias": torch.Size([32]),
            "segnet.asymmetric2_7.ext_conv3.conv2d.weight": torch.Size([128, 32, 1, 1]),
            "segnet.asymmetric2_7.ext_conv3.bn1.weight": torch.Size([128]),
            "segnet.asymmetric2_7.ext_conv3.bn1.bias": torch.Size([128]),
            "segnet.dilated2_8.ext_conv1.conv2d.weight": torch.Size([32, 128, 1, 1]),
            "segnet.dilated2_8.ext_conv1.bn1.weight": torch.Size([32]),
            "segnet.dilated2_8.ext_conv1.bn1.bias": torch.Size([32]),
            "segnet.dilated2_8.ext_conv2.conv2d.weight": torch.Size([32, 32, 3, 3]),
            "segnet.dilated2_8.ext_conv2.bn1.weight": torch.Size([32]),
            "segnet.dilated2_8.ext_conv2.bn1.bias": torch.Size([32]),
            "segnet.dilated2_8.ext_conv3.conv2d.weight": torch.Size([128, 32, 1, 1]),
            "segnet.dilated2_8.ext_conv3.bn1.weight": torch.Size([128]),
            "segnet.dilated2_8.ext_conv3.bn1.bias": torch.Size([128]),
            "segnet.regular3_0.ext_conv1.conv2d.weight": torch.Size([32, 128, 1, 1]),
            "segnet.regular3_0.ext_conv1.bn1.weight": torch.Size([32]),
            "segnet.regular3_0.ext_conv1.bn1.bias": torch.Size([32]),
            "segnet.regular3_0.ext_conv2.conv2d.weight": torch.Size([32, 32, 3, 3]),
            "segnet.regular3_0.ext_conv2.bn1.weight": torch.Size([32]),
            "segnet.regular3_0.ext_conv2.bn1.bias": torch.Size([32]),
            "segnet.regular3_0.ext_conv3.conv2d.weight": torch.Size([128, 32, 1, 1]),
            "segnet.regular3_0.ext_conv3.bn1.weight": torch.Size([128]),
            "segnet.regular3_0.ext_conv3.bn1.bias": torch.Size([128]),
            "segnet.dilated3_1.ext_conv1.conv2d.weight": torch.Size([32, 128, 1, 1]),
            "segnet.dilated3_1.ext_conv1.bn1.weight": torch.Size([32]),
            "segnet.dilated3_1.ext_conv1.bn1.bias": torch.Size([32]),
            "segnet.dilated3_1.ext_conv2.conv2d.weight": torch.Size([32, 32, 3, 3]),
            "segnet.dilated3_1.ext_conv2.bn1.weight": torch.Size([32]),
            "segnet.dilated3_1.ext_conv2.bn1.bias": torch.Size([32]),
            "segnet.dilated3_1.ext_conv3.conv2d.weight": torch.Size([128, 32, 1, 1]),
            "segnet.dilated3_1.ext_conv3.bn1.weight": torch.Size([128]),
            "segnet.dilated3_1.ext_conv3.bn1.bias": torch.Size([128]),
            "segnet.asymmetric3_2.ext_conv1.conv2d.weight": torch.Size([32, 128, 1, 1]),
            "segnet.asymmetric3_2.ext_conv1.bn1.weight": torch.Size([32]),
            "segnet.asymmetric3_2.ext_conv1.bn1.bias": torch.Size([32]),
            "segnet.asymmetric3_2.ext_conv2.conv2d1.weight": torch.Size([32, 32, 5, 1]),
            "segnet.asymmetric3_2.ext_conv2.bn1.weight": torch.Size([32]),
            "segnet.asymmetric3_2.ext_conv2.bn1.bias": torch.Size([32]),
            "segnet.asymmetric3_2.ext_conv2.conv2d2.weight": torch.Size([32, 32, 1, 5]),
            "segnet.asymmetric3_2.ext_conv2.bn2.weight": torch.Size([32]),
            "segnet.asymmetric3_2.ext_conv2.bn2.bias": torch.Size([32]),
            "segnet.asymmetric3_2.ext_conv3.conv2d.weight": torch.Size([128, 32, 1, 1]),
            "segnet.asymmetric3_2.ext_conv3.bn1.weight": torch.Size([128]),
            "segnet.asymmetric3_2.ext_conv3.bn1.bias": torch.Size([128]),
            "segnet.dilated3_3.ext_conv1.conv2d.weight": torch.Size([32, 128, 1, 1]),
            "segnet.dilated3_3.ext_conv1.bn1.weight": torch.Size([32]),
            "segnet.dilated3_3.ext_conv1.bn1.bias": torch.Size([32]),
            "segnet.dilated3_3.ext_conv2.conv2d.weight": torch.Size([32, 32, 3, 3]),
            "segnet.dilated3_3.ext_conv2.bn1.weight": torch.Size([32]),
            "segnet.dilated3_3.ext_conv2.bn1.bias": torch.Size([32]),
            "segnet.dilated3_3.ext_conv3.conv2d.weight": torch.Size([128, 32, 1, 1]),
            "segnet.dilated3_3.ext_conv3.bn1.weight": torch.Size([128]),
            "segnet.dilated3_3.ext_conv3.bn1.bias": torch.Size([128]),
            "segnet.regular3_4.ext_conv1.conv2d.weight": torch.Size([32, 128, 1, 1]),
            "segnet.regular3_4.ext_conv1.bn1.weight": torch.Size([32]),
            "segnet.regular3_4.ext_conv1.bn1.bias": torch.Size([32]),
            "segnet.regular3_4.ext_conv2.conv2d.weight": torch.Size([32, 32, 3, 3]),
            "segnet.regular3_4.ext_conv2.bn1.weight": torch.Size([32]),
            "segnet.regular3_4.ext_conv2.bn1.bias": torch.Size([32]),
            "segnet.regular3_4.ext_conv3.conv2d.weight": torch.Size([128, 32, 1, 1]),
            "segnet.regular3_4.ext_conv3.bn1.weight": torch.Size([128]),
            "segnet.regular3_4.ext_conv3.bn1.bias": torch.Size([128]),
            "segnet.dilated3_5.ext_conv1.conv2d.weight": torch.Size([32, 128, 1, 1]),
            "segnet.dilated3_5.ext_conv1.bn1.weight": torch.Size([32]),
            "segnet.dilated3_5.ext_conv1.bn1.bias": torch.Size([32]),
            "segnet.dilated3_5.ext_conv2.conv2d.weight": torch.Size([32, 32, 3, 3]),
            "segnet.dilated3_5.ext_conv2.bn1.weight": torch.Size([32]),
            "segnet.dilated3_5.ext_conv2.bn1.bias": torch.Size([32]),
            "segnet.dilated3_5.ext_conv3.conv2d.weight": torch.Size([128, 32, 1, 1]),
            "segnet.dilated3_5.ext_conv3.bn1.weight": torch.Size([128]),
            "segnet.dilated3_5.ext_conv3.bn1.bias": torch.Size([128]),
            "segnet.asymmetric3_6.ext_conv1.conv2d.weight": torch.Size([32, 128, 1, 1]),
            "segnet.asymmetric3_6.ext_conv1.bn1.weight": torch.Size([32]),
            "segnet.asymmetric3_6.ext_conv1.bn1.bias": torch.Size([32]),
            "segnet.asymmetric3_6.ext_conv2.conv2d1.weight": torch.Size([32, 32, 5, 1]),
            "segnet.asymmetric3_6.ext_conv2.bn1.weight": torch.Size([32]),
            "segnet.asymmetric3_6.ext_conv2.bn1.bias": torch.Size([32]),
            "segnet.asymmetric3_6.ext_conv2.conv2d2.weight": torch.Size([32, 32, 1, 5]),
            "segnet.asymmetric3_6.ext_conv2.bn2.weight": torch.Size([32]),
            "segnet.asymmetric3_6.ext_conv2.bn2.bias": torch.Size([32]),
            "segnet.asymmetric3_6.ext_conv3.conv2d.weight": torch.Size([128, 32, 1, 1]),
            "segnet.asymmetric3_6.ext_conv3.bn1.weight": torch.Size([128]),
            "segnet.asymmetric3_6.ext_conv3.bn1.bias": torch.Size([128]),
            "segnet.dilated3_7.ext_conv1.conv2d.weight": torch.Size([32, 128, 1, 1]),
            "segnet.dilated3_7.ext_conv1.bn1.weight": torch.Size([32]),
            "segnet.dilated3_7.ext_conv1.bn1.bias": torch.Size([32]),
            "segnet.dilated3_7.ext_conv2.conv2d.weight": torch.Size([32, 32, 3, 3]),
            "segnet.dilated3_7.ext_conv2.bn1.weight": torch.Size([32]),
            "segnet.dilated3_7.ext_conv2.bn1.bias": torch.Size([32]),
            "segnet.dilated3_7.ext_conv3.conv2d.weight": torch.Size([128, 32, 1, 1]),
            "segnet.dilated3_7.ext_conv3.bn1.weight": torch.Size([128]),
            "segnet.dilated3_7.ext_conv3.bn1.bias": torch.Size([128]),
            "segnet.upsample4_0.main_conv1.conv2d.weight": torch.Size([64, 128, 1, 1]),
            "segnet.upsample4_0.main_conv1.bn1.weight": torch.Size([64]),
            "segnet.upsample4_0.main_conv1.bn1.bias": torch.Size([64]),
            "segnet.upsample4_0.ext_conv1.conv2d.weight": torch.Size([32, 128, 1,1]),
            "segnet.upsample4_0.ext_conv1.bn1.weight": torch.Size([32]),
            "segnet.upsample4_0.ext_conv1.bn1.bias": torch.Size([32]),
            "segnet.upsample4_0.ext_tconv1.conv2d.weight": torch.Size([32, 32, 2,2]),
            "segnet.upsample4_0.ext_tconv1.bn1.weight": torch.Size([32]),
            "segnet.upsample4_0.ext_tconv1.bn1.bias": torch.Size([32]),
            "segnet.upsample4_0.ext_conv2.conv2d.weight": torch.Size([64, 32, 1,1]),
            "segnet.upsample4_0.ext_conv2.bn1.weight": torch.Size([64]),
            "segnet.upsample4_0.ext_conv2.bn1.bias": torch.Size([64]),
            "segnet.regular4_1.ext_conv1.conv2d.weight": torch.Size([16, 64, 1, 1]),
            "segnet.regular4_1.ext_conv1.bn1.weight": torch.Size([16]),
            "segnet.regular4_1.ext_conv1.bn1.bias": torch.Size([16]),
            "segnet.regular4_1.ext_conv2.conv2d.weight": torch.Size([16, 16, 3, 3]),
            "segnet.regular4_1.ext_conv2.bn1.weight": torch.Size([16]),
            "segnet.regular4_1.ext_conv2.bn1.bias": torch.Size([16]),
            "segnet.regular4_1.ext_conv3.conv2d.weight": torch.Size([64, 16, 1, 1]),
            "segnet.regular4_1.ext_conv3.bn1.weight": torch.Size([64]),
            "segnet.regular4_1.ext_conv3.bn1.bias": torch.Size([64]),
            "segnet.regular4_2.ext_conv1.conv2d.weight": torch.Size([16, 64, 1, 1]),
            "segnet.regular4_2.ext_conv1.bn1.weight": torch.Size([16]),
            "segnet.regular4_2.ext_conv1.bn1.bias": torch.Size([16]),
            "segnet.regular4_2.ext_conv2.conv2d.weight": torch.Size([16, 16, 3, 3]),
            "segnet.regular4_2.ext_conv2.bn1.weight": torch.Size([16]),
            "segnet.regular4_2.ext_conv2.bn1.bias": torch.Size([16]),
            "segnet.regular4_2.ext_conv3.conv2d.weight": torch.Size([64, 16, 1, 1]),
            "segnet.regular4_2.ext_conv3.bn1.weight": torch.Size([64]),
            "segnet.regular4_2.ext_conv3.bn1.bias": torch.Size([64]),
            "segnet.upsample5_0.main_conv1.conv2d.weight": torch.Size([16, 64, 1, 1]),
            "segnet.upsample5_0.main_conv1.bn1.weight": torch.Size([16]),
            "segnet.upsample5_0.main_conv1.bn1.bias": torch.Size([16]),
            "segnet.upsample5_0.ext_conv1.conv2d.weight": torch.Size([16, 64, 1,1]),
            "segnet.upsample5_0.ext_conv1.bn1.weight": torch.Size([16]),
            "segnet.upsample5_0.ext_conv1.bn1.bias": torch.Size([16]),
            "segnet.upsample5_0.ext_tconv1.conv2d.weight": torch.Size([16, 16, 2,2]),
            "segnet.upsample5_0.ext_tconv1.bn1.weight": torch.Size([16]),
            "segnet.upsample5_0.ext_tconv1.bn1.bias": torch.Size([16]),
            "segnet.upsample5_0.ext_conv2.conv2d.weight": torch.Size([16, 16, 1,1]),
            "segnet.upsample5_0.ext_conv2.bn1.weight": torch.Size([16]),
            "segnet.upsample5_0.ext_conv2.bn1.bias": torch.Size([16]),    
            "segnet.regular5_1.ext_conv1.conv2d.weight": torch.Size([4, 16, 1, 1]),
            "segnet.regular5_1.ext_conv1.bn1.weight": torch.Size([4]),
            "segnet.regular5_1.ext_conv1.bn1.bias": torch.Size([4]),
            "segnet.regular5_1.ext_conv2.conv2d.weight": torch.Size([4, 4, 3, 3]),
            "segnet.regular5_1.ext_conv2.bn1.weight": torch.Size([4]),
            "segnet.regular5_1.ext_conv2.bn1.bias": torch.Size([4]),
            "segnet.regular5_1.ext_conv3.conv2d.weight": torch.Size([16, 4, 1, 1]),
            "segnet.regular5_1.ext_conv3.bn1.weight": torch.Size([16]),
            "segnet.regular5_1.ext_conv3.bn1.bias": torch.Size([16]),        
            #"segnet.pred_task1.conv2d2.weight": torch.Size([64, 64, 3, 3]),
            "segnet.pred_task1.transposed_conv1.weight": torch.Size([16, 13, 3, 3]),
            "segnet.pred_task1.transposed_conv2.weight": torch.Size([16, 13, 2, 2]),
            #"segnet.pred_task2.conv2d1.weight": torch.Size([64, 64, 3, 3]),
            "segnet.pred_task2.transposed_conv1.weight": torch.Size([16, 1, 3, 3]),
            "segnet.pred_task2.transposed_conv2.weight": torch.Size([16, 1, 2, 2]),
            #"segnet.pred_task3.conv2d1.weight": torch.Size([64, 64, 3, 3]),
            "segnet.pred_task3.transposed_conv1.weight": torch.Size([16, 3, 3, 3]),
            "segnet.pred_task3.transposed_conv2.weight": torch.Size([16, 3, 2, 2]),
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


class ENetTarget(nn.Module):
    def __init__(self, pretrained=False, progress=True,momentum=None,model_type="standard", n_tasks = None,**kwargs):
        super().__init__()
        self.momentum = momentum
        self.n_tasks = n_tasks
        self.model_type = model_type
    def forward(self, x, weights=None):
        device = x.device
        # Initial block
        a = torch.tensor([[1]]).float().to(device)
        input_size = x.size()
        x = self.initial_block(x,weights,a)
        p=0.01
        # Stage 1 - Encoder
        stage1_input_size = x.size()
        x, max_indices1_0 = self.downsample(x,weights,1,p,a)
        x = self.regular(x, weights,1,1,p,a) 
        x = self.regular(x, weights,1,2,p,a)
        x = self.regular(x, weights,1,3,p,a)
        x = self.regular(x, weights,1,4,p,a)

        # Stage 2 - Encoder
        stage2_input_size = x.size()
        #print("stage2_input_size: ",stage2_input_size)
        x, max_indices2_0 = self.downsample(x,weights,2,p,a)
        p=0.1
        x = self.regular(x, weights,2,1,p,a) 
        x = self.dilated(x, weights,2,2,0.1,2,2,a)
        x = self.asymmetric(x, weights,2,3,p,a) 
        x = self.dilated(x, weights,2,4,0.1,4,4,a)
        x = self.regular(x, weights,2,5,p,a) 
        x = self.dilated(x, weights,2,6,0.1,8,8,a)
        x = self.asymmetric(x, weights,2,7,p,a) 
        x = self.dilated(x, weights,2,8,0.1,16,16,a)

        # Stage 3 - Encoder
        x = self.regular(x, weights,3,0,p,a) 
        x = self.dilated(x, weights,3,1,0.1,2,2,a)
        x = self.asymmetric(x, weights,3,2,p,a)
        x = self.dilated(x, weights,3,3,0.1,4,4,a)
        x = self.regular(x, weights,3,4,p,a) 
        x = self.dilated(x, weights,3,5,0.1,8,8,a)
        x = self.asymmetric(x, weights,3,6,p,a)
        x = self.dilated(x, weights,3,7,0.1,16,16,a)

        # Stage 4 - Decoder
        #print(max_indices2_0.shape)
        #print(stage1_input_size)
        x = self.upsample(x, weights,4,0.1,max_indices2_0)
        x = self.regular(x, weights,4,1,p,a) 
        x = self.regular(x, weights,4,2,p,a) 

        # Stage 5 - Decoder
        x = self.upsample(x, weights,5,0.1,max_indices1_0)
        x = self.regular(x, weights,5,1,p,a) 
        #print(x.shape)
        #print(self.forward_clf(x,weights,1).shape)
        #print(x.shape)
        t1_pred = F.log_softmax(self.forward_clf(x,weights,1,input_size), dim=1)
        #print(t1_pred.shape)
        t2_pred = self.forward_clf(x,weights,2,input_size)
        t3_pred = self.forward_clf(x,weights,3,input_size)
        t3_pred = t3_pred / torch.norm(t3_pred, p=2, dim=1, keepdim=True)

        return [t1_pred, t2_pred, t3_pred]

    def initial_block(self,x,weights,a):
        device = x.device
        main = F.conv2d(x, weights["segnet.initial_block.conv2d.weight"], stride=2, padding=1)
        x = F.max_pool2d(x, kernel_size=3, stride=2,padding = 1, dilation=1, ceil_mode=False)
        # Concatenate branches
        #print(x.shape,main.shape)
        x = torch.cat((main, x), 1)
        #print(x.shape)
        x = F.batch_norm(
            x,
            torch.zeros(x.data.size()[1]).to(device),
            torch.ones(x.data.size()[1]).to(device),
            weights["segnet.initial_block.bn1.weight"],
            weights["segnet.initial_block.bn1.bias"],
            training=True,
        )
       
        x = F.prelu(x,weight=a) 
        return x
    def downsample(self,x, weights,index,p,a):
        """Before blocks"""
        """
        "segnet.downsample1_0.ext_conv1.conv2d.weight": torch.Size([4, 16, 2, 2]),
        "segnet.downsample1_0.ext_conv1.bn1.weight": torch.Size([4]),
        "segnet.downsample1_0.ext_conv1.bn1.bias": torch.Size([4]),
        "segnet.downsample1_0.ext_conv2.conv2d.weight": torch.Size([4, 4, 3,3]),
        "segnet.downsample1_0.ext_conv2.bn1.weight": torch.Size([4]),
        "segnet.downsample1_0.ext_conv2.bn1.bias": torch.Size([4]),
        "segnet.downsample1_0.ext_conv3.conv2d.weight": torch.Size([64, 4, 1,1]),
        "segnet.downsample1_0.ext_conv3.bn1.weight": torch.Size([64]),
        "segnet.downsample1_0.ext_conv3.bn1.bias": torch.Size([64]),
        """
        '''
            "segnet.downsample2_0.ext_conv1.conv2d.weight": torch.Size([16, 64, 2, 2]),
            "segnet.downsample2_0.ext_conv1.bn1.weight": torch.Size([16]),
            "segnet.downsample2_0.ext_conv1.bn1.bias": torch.Size([16]),
            "segnet.downsample2_0.ext_conv2.conv2d.weight": torch.Size([16, 16, 3,3]),
            "segnet.downsample2_0.ext_conv2.bn1.weight": torch.Size([16]),
            "segnet.downsample2_0.ext_conv2.bn1.bias": torch.Size([16]),
            "segnet.downsample2_0.ext_conv3.conv2d.weight": torch.Size([128, 16, 1,1]),
            "segnet.downsample2_0.ext_conv3.bn1.weight": torch.Size([128]),
            "segnet.downsample2_0.ext_conv3.bn1.bias": torch.Size([128]),
        '''

        device = x.device
        main,max_indices = F.max_pool2d(x, kernel_size=2, stride=2,padding = 0, dilation=1, ceil_mode=False,return_indices=True)
        # conv1
        x = F.conv2d(x, weights[f"segnet.downsample{index}_0.ext_conv1.conv2d.weight"], stride=2)
        # bn1
        x = F.batch_norm(
            x,
            torch.zeros(x.data.size()[1]).to(device),
            torch.ones(x.data.size()[1]).to(device),
            weights[f"segnet.downsample{index}_0.ext_conv1.bn1.weight"],
            weights[f"segnet.downsample{index}_0.ext_conv1.bn1.bias"],
            training=True,
        )
        x = F.prelu(x,weight=a)
        # conv2
        x = F.conv2d(x, weights[f"segnet.downsample{index}_0.ext_conv2.conv2d.weight"], stride=1, padding=1)
        # bn2
        x = F.batch_norm(
            x,
            torch.zeros(x.data.size()[1]).to(device),
            torch.ones(x.data.size()[1]).to(device),
            weights[f"segnet.downsample{index}_0.ext_conv2.bn1.weight"],
            weights[f"segnet.downsample{index}_0.ext_conv2.bn1.bias"],
            training=True,
        )
        x = F.prelu(x,weight=a)
        # conv3
        x = F.conv2d(x, weights[f"segnet.downsample{index}_0.ext_conv3.conv2d.weight"], stride=1)
        # bn3
        x = F.batch_norm(
            x,
            torch.zeros(x.data.size()[1]).to(device),
            torch.ones(x.data.size()[1]).to(device),
            weights[f"segnet.downsample{index}_0.ext_conv3.bn1.weight"],
            weights[f"segnet.downsample{index}_0.ext_conv3.bn1.bias"],
            training=True,
        )
        x = F.prelu(x,weight=a)
        x = F.dropout(x,p=p)
        # Main branch channel padding
        n, ch_ext, h, w = x.size()
        ch_main = main.size()[1]
        padding = torch.zeros(n, ch_ext - ch_main, h, w)

        # Before concatenating, check if main is on the CPU or GPU and
        # convert padding accordingly
        
        padding = padding.to(device)

        # Concatenate
        main = torch.cat((main, padding), 1)

        # Add main and extension branches
        out = main + x
        x = F.prelu(out,weight=a)
        return x,max_indices
    def upsample(self,x, weights,index,p,max_indices):
        """Before blocks"""
        """
            "segnet.upsample4_0.main_conv1.conv2d.weight": torch.Size([64, 128, 1, 1]),
            "segnet.upsample4_0.main_conv1.bn1.weight": torch.Size([64]),
            "segnet.upsample4_0.main_conv1.bn1.bias": torch.Size([64]),
            "segnet.upsample4_0.ext_conv1.conv2d.weight": torch.Size([32, 128, 1,1]),
            "segnet.upsample4_0.ext_conv1.bn1.weight": torch.Size([32]),
            "segnet.upsample4_0.ext_conv1.bn1.bias": torch.Size([32]),
            "segnet.upsample4_0.ext_tconv1.conv2d.weight": torch.Size([32, 32, 2,2]),
            "segnet.upsample4_0.ext_tconv1.bn1.weight": torch.Size([32]),
            "segnet.upsample4_0.ext_tconv1.bn1.bias": torch.Size([32]),
            "segnet.upsample4_0.ext_conv2.conv2d.weight": torch.Size([64, 32, 1,1]),
            "segnet.upsample4_0.ext_conv2.bn1.weight": torch.Size([64]),
            "segnet.upsample4_0.ext_conv2.bn1.bias": torch.Size([64]),
        """
        '''
            "segnet.upsample5_0.main_conv1.conv2d.weight": torch.Size([16, 64, 1, 1]),
            "segnet.upsample5_0.main_conv1.bn1.weight": torch.Size([16]),
            "segnet.upsample5_0.main_conv1.bn1.bias": torch.Size([16]),
            "segnet.upsample5_0.ext_conv1.conv2d.weight": torch.Size([16, 64, 1,1]),
            "segnet.upsample5_0.ext_conv1.bn1.weight": torch.Size([16]),
            "segnet.upsample5_0.ext_conv1.bn1.bias": torch.Size([16]),
            "segnet.upsample5_0.ext_tconv1.conv2d.weight": torch.Size([16, 16, 2,2]),
            "segnet.upsample5_0.ext_tconv1.bn1.weight": torch.Size([16]),
            "segnet.upsample5_0.ext_tconv1.bn1.bias": torch.Size([16]),
            "segnet.upsample5_0.ext_conv2.conv2d.weight": torch.Size([16, 16, 1,1]),
            "segnet.upsample5_0.ext_conv2.bn1.weight": torch.Size([16]),
            "segnet.upsample5_0.ext_conv2.bn1.bias": torch.Size([16]),    
        '''

        device = x.device
        # mainconv
        main = F.conv2d(x, weights[f"segnet.upsample{index}_0.main_conv1.conv2d.weight"], stride=1)
        main = F.batch_norm(
            main,
            torch.zeros(main.data.size()[1]).to(device),
            torch.ones(main.data.size()[1]).to(device),
            weights[f"segnet.upsample{index}_0.main_conv1.bn1.weight"],
            weights[f"segnet.upsample{index}_0.main_conv1.bn1.bias"],
            training=True,
        )
        
        #print(main.shape,max_indices.shape)
        main = F.max_unpool2d(main,indices = max_indices, kernel_size=2, stride=2,padding = 0)
        #print(main.shape)
        # conv1
        x = F.conv2d(x, weights[f"segnet.upsample{index}_0.ext_conv1.conv2d.weight"], stride=1)
        # bn1
        x = F.batch_norm(
            x,
            torch.zeros(x.data.size()[1]).to(device),
            torch.ones(x.data.size()[1]).to(device),
            weights[f"segnet.upsample{index}_0.ext_conv1.bn1.weight"],
            weights[f"segnet.upsample{index}_0.ext_conv1.bn1.bias"],
            training=True,
        )
        x = F.relu(x)
        # tconv
        x = F.conv_transpose2d(x, weights[f"segnet.upsample{index}_0.ext_tconv1.conv2d.weight"], stride=2)
        
        x = F.batch_norm(
            x,
            torch.zeros(x.data.size()[1]).to(device),
            torch.ones(x.data.size()[1]).to(device),
            weights[f"segnet.upsample{index}_0.ext_tconv1.bn1.weight"],
            weights[f"segnet.upsample{index}_0.ext_tconv1.bn1.bias"],
            training=True,
        )
        
        x = F.relu(x)
        # conv2
        x = F.conv2d(x, weights[f"segnet.upsample{index}_0.ext_conv2.conv2d.weight"], stride=1)
        #print(x.shape)
        # bn2
        x = F.batch_norm(
            x,
            torch.zeros(x.data.size()[1]).to(device),
            torch.ones(x.data.size()[1]).to(device),
            weights[f"segnet.upsample{index}_0.ext_conv2.bn1.weight"],
            weights[f"segnet.upsample{index}_0.ext_conv2.bn1.bias"],
            training=True,
        )
        x = F.dropout(x,p=p)
        #print(x.shape)
        x = main + x
        x = F.relu(x)
        return x
    def regular(self,x, weights,block,index,p,a):
        """Before blocks"""
        """
            "segnet.regular1_1.ext_conv1.conv2d.weight": torch.Size([16, 64, 1, 1]),
            "segnet.regular1_1.ext_conv1.bn1.weight": torch.Size([16]),
            "segnet.regular1_1.ext_conv1.bn1.bias": torch.Size([16]),
            "segnet.regular1_1.ext_conv2.conv2d.weight": torch.Size([16, 16, 3, 3]),
            "segnet.regular1_1.ext_conv2.bn1.weight": torch.Size([16]),
            "segnet.regular1_1.ext_conv2.bn1.bias": torch.Size([16]),
            "segnet.regular1_1.ext_conv3.conv2d.weight": torch.Size([64, 16, 1, 1]),
            "segnet.regular1_1.ext_conv3.bn1.weight": torch.Size([64]),
            "segnet.regular1_1.ext_conv3.bn1.bias": torch.Size([64]),
        """

        device = x.device
        main = x
        # conv1
        x = F.conv2d(x, weights[f"segnet.regular{block}_{index}.ext_conv1.conv2d.weight"], stride=1)
        # bn1
        x = F.batch_norm(
            x,
            torch.zeros(x.data.size()[1]).to(device),
            torch.ones(x.data.size()[1]).to(device),
            weights[f"segnet.regular{block}_{index}.ext_conv1.bn1.weight"],
            weights[f"segnet.regular{block}_{index}.ext_conv1.bn1.bias"],
            training=True,
        )
        x = F.prelu(x,weight=a)
        # conv2
        x = F.conv2d(x, weights[f"segnet.regular{block}_{index}.ext_conv2.conv2d.weight"], stride=1, padding=1)
        # bn2
        x = F.batch_norm(
            x,
            torch.zeros(x.data.size()[1]).to(device),
            torch.ones(x.data.size()[1]).to(device),
            weights[f"segnet.regular{block}_{index}.ext_conv2.bn1.weight"],
            weights[f"segnet.regular{block}_{index}.ext_conv2.bn1.bias"],
            training=True,
        )
        x = F.prelu(x,weight=a)
        # conv3
        x = F.conv2d(x, weights[f"segnet.regular{block}_{index}.ext_conv3.conv2d.weight"], stride=1)
        # bn3
        x = F.batch_norm(
            x,
            torch.zeros(x.data.size()[1]).to(device),
            torch.ones(x.data.size()[1]).to(device),
            weights[f"segnet.regular{block}_{index}.ext_conv3.bn1.weight"],
            weights[f"segnet.regular{block}_{index}.ext_conv3.bn1.bias"],
            training=True,
        )
        x = F.prelu(x,weight=a)
        x = F.dropout(x,p=p)
        x = main + x
        x = F.prelu(x,weight=a)
        return x
    def dilated(self,x, weights,block,index,p,padding,dilation,a):
        """Before blocks"""
        """
            "segnet.dilated2_2.ext_conv1.conv2d.weight": torch.Size([32, 128, 1, 1]),
            "segnet.dilated2_2.ext_conv1.bn1.weight": torch.Size([32]),
            "segnet.dilated2_2.ext_conv1.bn1.bias": torch.Size([32]),
            "segnet.dilated2_2.ext_conv2.conv2d.weight": torch.Size([32, 32, 3, 3]),
            "segnet.dilated2_2.ext_conv2.bn1.weight": torch.Size([32]),
            "segnet.dilated2_2.ext_conv2.bn1.bias": torch.Size([32]),
            "segnet.dilated2_2.ext_conv3.conv2d.weight": torch.Size([128, 32, 1, 1]),
            "segnet.dilated2_2.ext_conv3.bn1.weight": torch.Size([128]),
            "segnet.dilated2_2.ext_conv3.bn1.bias": torch.Size([128]),
        """

        device = x.device
        main = x
        # conv1
        x = F.conv2d(x, weights[f"segnet.dilated{block}_{index}.ext_conv1.conv2d.weight"], stride=1)
        # bn1
        x = F.batch_norm(
            x,
            torch.zeros(x.data.size()[1]).to(device),
            torch.ones(x.data.size()[1]).to(device),
            weights[f"segnet.dilated{block}_{index}.ext_conv1.bn1.weight"],
            weights[f"segnet.dilated{block}_{index}.ext_conv1.bn1.bias"],
            training=True,
        )
        x = F.prelu(x,weight=a)
        # conv2
        x = F.conv2d(x, weights[f"segnet.dilated{block}_{index}.ext_conv2.conv2d.weight"], stride=1, padding=padding,dilation=dilation)
        # bn2
        x = F.batch_norm(
            x,
            torch.zeros(x.data.size()[1]).to(device),
            torch.ones(x.data.size()[1]).to(device),
            weights[f"segnet.dilated{block}_{index}.ext_conv2.bn1.weight"],
            weights[f"segnet.dilated{block}_{index}.ext_conv2.bn1.bias"],
            training=True,
        )
        x = F.prelu(x,weight=a)
        # conv3
        x = F.conv2d(x, weights[f"segnet.dilated{block}_{index}.ext_conv3.conv2d.weight"], stride=1)
        # bn3
        x = F.batch_norm(
            x,
            torch.zeros(x.data.size()[1]).to(device),
            torch.ones(x.data.size()[1]).to(device),
            weights[f"segnet.dilated{block}_{index}.ext_conv3.bn1.weight"],
            weights[f"segnet.dilated{block}_{index}.ext_conv3.bn1.bias"],
            training=True,
        )
        x = F.prelu(x,weight=a)
        x = F.dropout(x,p=p)
        x = main + x
        x = F.prelu(x,weight=a)
        return x
    def asymmetric(self,x, weights,block,index,p,a):
        """Before blocks"""
        """
            "segnet.asymmetric2_3.ext_conv1.conv2d.weight": torch.Size([32, 128, 1, 1]),
            "segnet.asymmetric2_3.ext_conv1.bn1.weight": torch.Size([32]),
            "segnet.asymmetric2_3.ext_conv1.bn1.bias": torch.Size([32]),
            "segnet.asymmetric2_3.ext_conv2.conv2d1.weight": torch.Size([32, 32, 5, 1]),
            "segnet.asymmetric2_3.ext_conv2.bn1.weight": torch.Size([32]),
            "segnet.asymmetric2_3.ext_conv2.bn1.bias": torch.Size([32]),
            "segnet.asymmetric2_3.ext_conv2.conv2d2.weight": torch.Size([32, 32, 1, 5]),
            "segnet.asymmetric2_3.ext_conv2.bn2.weight": torch.Size([32]),
            "segnet.asymmetric2_3.ext_conv2.bn2.bias": torch.Size([32]),
            "segnet.asymmetric2_3.ext_conv3.conv2d.weight": torch.Size([128, 32, 1, 1]),
            "segnet.asymmetric2_3.ext_conv3.bn1.weight": torch.Size([128]),
            "segnet.asymmetric2_3.ext_conv3.bn1.bias": torch.Size([128]),
        """

        device = x.device
        main = x
        # conv1
        x = F.conv2d(x, weights[f"segnet.asymmetric{block}_{index}.ext_conv1.conv2d.weight"], stride=1)
        # bn1
        x = F.batch_norm(
            x,
            torch.zeros(x.data.size()[1]).to(device),
            torch.ones(x.data.size()[1]).to(device),
            weights[f"segnet.asymmetric{block}_{index}.ext_conv1.bn1.weight"],
            weights[f"segnet.asymmetric{block}_{index}.ext_conv1.bn1.bias"],
            training=True,
        )
        x = F.prelu(x,weight=a)
        # conv2.1
        x = F.conv2d(x, weights[f"segnet.asymmetric{block}_{index}.ext_conv2.conv2d1.weight"], stride=1, padding=(2,0))
        # bn2.1
        x = F.batch_norm(
            x,
            torch.zeros(x.data.size()[1]).to(device),
            torch.ones(x.data.size()[1]).to(device),
            weights[f"segnet.asymmetric{block}_{index}.ext_conv2.bn1.weight"],
            weights[f"segnet.asymmetric{block}_{index}.ext_conv2.bn1.bias"],
            training=True,
        )
        x = F.prelu(x,weight=a)
        # conv2.2
        x = F.conv2d(x, weights[f"segnet.asymmetric{block}_{index}.ext_conv2.conv2d2.weight"], stride=1, padding=(0,2))
        # bn2.2
        x = F.batch_norm(
            x,
            torch.zeros(x.data.size()[1]).to(device),
            torch.ones(x.data.size()[1]).to(device),
            weights[f"segnet.asymmetric{block}_{index}.ext_conv2.bn2.weight"],
            weights[f"segnet.asymmetric{block}_{index}.ext_conv2.bn2.bias"],
            training=True,
        )
        x = F.prelu(x,weight=a)
        # conv3
        x = F.conv2d(x, weights[f"segnet.asymmetric{block}_{index}.ext_conv3.conv2d.weight"], stride=1)
        # bn3
        x = F.batch_norm(
            x,
            torch.zeros(x.data.size()[1]).to(device),
            torch.ones(x.data.size()[1]).to(device),
            weights[f"segnet.asymmetric{block}_{index}.ext_conv3.bn1.weight"],
            weights[f"segnet.asymmetric{block}_{index}.ext_conv3.bn1.bias"],
            training=True,
        )
        x = F.prelu(x,weight=a)
        x = F.dropout(x,p=p)
        x = main + x
        x = F.prelu(x,weight=a)
        return x
    def forward_clf(self,x, weights, index,input_size):
        #x = F.conv_transpose2d(x, weights[f"segnet.pred_task{index}.transposed_conv1.weight"], stride=2, padding=0)
        x = F.conv_transpose2d(x, weights[f"segnet.pred_task{index}.transposed_conv1.weight"], stride=2, padding=1,output_padding=1)
        return x