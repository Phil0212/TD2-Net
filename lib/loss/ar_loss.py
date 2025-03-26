import torch
import torch.nn as nn
import pandas as np
import pickle


class AsymmetricLoss_ar(nn.Module):

    def __init__(self, config=None, bce_weights=None, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, beta=0.9999,
                 gamma=0.5, disable_torch_grad_focal_loss=True, omega=True):
        super(AsymmetricLoss_ar, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps
        self.omega = omega

    def forward(self, x, y, class_weight):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        # Calculating Probabilities
        # x_sigmoid = torch.sigmoid(x)

        xs_pos = x
        xs_neg = 1 - x

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic CE calculation
        if self.omega:
            los_pos = torch.mul(class_weight.cuda(), y * torch.log(xs_pos.clamp(min=self.eps))) * 10000
        else:
            los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
            
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg
   
        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)  # pt = p if t > 0 else 1-p
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            loss *= one_sided_w

        return -(loss.sum() / x.shape[0])


