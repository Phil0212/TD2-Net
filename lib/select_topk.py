import torch
import torch.nn as nn
import torch.nn.functional as F


class Selector(nn.Module):
    def __init__(self, topk, selection_method='gumbel', obj_dim=2048, dim=512):
        super(Selector, self).__init__()
        self.linear_Q = nn.Linear(obj_dim, dim)
        self.norm_Q = nn.LayerNorm(dim, eps=1e-12)

        self.linear_K = nn.Linear(obj_dim, dim)
        self.norm_K = nn.LayerNorm(dim, eps=1e-12)

        self.topk = topk
        self.selection_method = selection_method

        self.fc = nn.Linear(self.topk*obj_dim, obj_dim)

    @staticmethod
    def sample_gumbel(n, k):
        unif = torch.distributions.Uniform(0, 1).sample((n, k))
        g = -torch.log(-torch.log(unif))
        return g

    # @staticmethod
    def sample_gumbel_softmax(self, pi, temperature):
        n, k = pi.shape
        # dbg.set_trace()
        g = self.sample_gumbel(n, k).to(pi.device)
        h = (g + torch.log(pi)) / temperature
        h_max = h.max(dim=1, keepdim=True)[0]
        h = h - h_max
        cache = torch.exp(h)
        #     print(pi, torch.log(pi), intmdt)
        y = cache / cache.sum(dim=-1, keepdim=True)
        return y

    def forward(self, Q, K, V):

        Q = self.norm_Q(self.linear_Q(Q))  
        K = self.norm_K(self.linear_K(K))  

        logit_scale = 1
        x_logits = logit_scale * Q @ K.t()
        x_logits = torch.softmax(x_logits, dim=-1)
        
        _segs = []
        for _ in range(self.topk):
            # selection_mask = self.sample_gumbel_softmax(x_logits, 1)
            selection_mask = F.gumbel_softmax(x_logits, tau=1, dim=-1)
            # if torch.isnan(selection_mask).sum() or torch.isinf(selection_mask).sum():
            #     print("----- help me ! ----")
            _segs.append(torch.matmul(selection_mask, V).unsqueeze(0))

        selected_segs = torch.flatten(torch.stack(_segs, dim=1).squeeze(0).permute(1, 0, 2), start_dim=1) 
      
        return self.fc(selected_segs)
