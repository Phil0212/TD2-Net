"""
Let's get the relationships yo
"""

import torch
import torch.nn as nn
import math

from torch.nn.utils.rnn import pad_sequence
from lib.word_vectors import obj_edge_vectors
from lib.transformer import transformer
from lib.fpn.box_utils import center_size
from fasterRCNN.lib.model.roi_layers import ROIAlign, nms
from lib.draw_rectangles.draw_rectangles import draw_union_boxes
from lib.select_topk import Selector
from lib.st_mp import TransformerEncoder, TransformerEncoderLayer


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 500):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
     
        position = torch.arange(max_len).unsqueeze(1).type(torch.float32).cuda()
        div_term = torch.exp((torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)).type(torch.float32).cuda())
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor, indices=None) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        if indices is None:
            x = x + self.pe[:,:x.size(1)]
        else:
            pos = torch.cat([self.pe[:, index] for index in indices])
            x = x + pos
        return self.dropout(x)


class DTrans(nn.Module):
    """
    Module for computing the object contexts and edge contexts
    """

    def __init__(self, mode='sgdet',d_model=2048, enc_layer_num=3, nhead=8, dropout=0.1, topk=8, encoder_layer=3, obj_classes=None):
        super(DTrans, self).__init__()
        self.classes = obj_classes
        self.topk = topk

        self.obj_selector = Selector(topk=self.topk, obj_dim=d_model)

        self.decoder_lin = nn.Sequential(nn.Linear(2048, 1024),
                                         nn.BatchNorm1d(1024),
                                         nn.ReLU(),
                                         nn.Linear(1024, len(self.classes)-1))

        self.positional_encoder = PositionalEncoding(d_model, 0.1, 600 if mode=="sgdet" else 400)
        encoder_layer = TransformerEncoderLayer(nhead=nhead, dropout=dropout)
        self.t_mp = TransformerEncoder(encoder_layer, enc_layer_num)
        self.s_mp = TransformerEncoder(encoder_layer, enc_layer_num)

        # self.adjecent_en = Local_Info(input_dims=2048)

        self.h2l = nn.Linear(2376, d_model)
        self.l2h = nn.Linear(d_model, 2048)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, entry):

        obj_features = entry['features']
        s_features = torch.tensor([]).cuda()

        for i in range( entry['features'].shape[0]):        
            
            pos_index = []
            indices = entry['match_node'][i].long()

            # index encoding
            im_idx, counts = torch.unique(entry["boxes"][indices][:, 0].view(-1), return_counts=True, sorted=True)
            counts = counts.tolist()
            pos = torch.cat([torch.LongTensor([im] * count) for im, count in zip(range(len(counts)), counts)])
            pos_index.append(pos)

            # features  
            sequence_features = pad_sequence([obj_features[index] for index in indices]).permute(1,0).unsqueeze(0)
            pos_index = pad_sequence(pos_index,  batch_first=True)

            match_features = self.positional_encoder(sequence_features , pos_index).squeeze(0)
            
            # select_topk
            select_topK = self.obj_selector(Q=entry['features'][i].unsqueeze(0), K=match_features, V=match_features)
            s_features = torch.cat((s_features, select_topK), dim=0) 

        # tempural message passing
        obj_ = self.t_mp(Q=entry['features'], K=s_features, V=s_features).squeeze(0)         
            
        # spatial message passing
        features = self.s_mp(Q=obj_, K=obj_, V=obj_).squeeze(0)    
      
        entry['features'] = features
        entry['distribution'] = self.decoder_lin(entry['features'])
        
        return entry


