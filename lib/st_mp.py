import torch
import torch.nn as nn
import copy

class TransformerEncoderLayer(nn.Module):

    def __init__(self, embed_dim=2048, nhead=4, dim_feedforward=1024, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, nhead, dropout=dropout)

        self.linear1 = nn.Linear(embed_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, embed_dim)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, Q, K, V, input_key_padding_mask=None):
        # local attention
        src = Q
        src2, local_attention_weights = self.self_attn(Q.unsqueeze(0), K.unsqueeze(0), V.unsqueeze(0), key_padding_mask=input_key_padding_mask)

        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(nn.functional.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        return src, local_attention_weights

class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, Q, K, V, input_key_padding_mask=None):
        
        for i, layer in enumerate(self.layers):
            output, local_attention_weights = layer(Q, K, V, input_key_padding_mask)
        
        if self.num_layers > 0:
            return output
        else:
            return output, None
        

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])