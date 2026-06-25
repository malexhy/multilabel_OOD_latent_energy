import torch
from torch import nn

def ASL(gamma_pos = 0.,
        gamma_neg = 4.,
        margin = 0.05,  
        mode = 'normal',
        dp_target = 0.1,
        ss = 0.01,
        ls = 0.,
        gamma_neg_min=1.,
        penalty_grad = False,
        ):
    
    if mode == 'normal':
        return ASL_normal(gamma_pos=gamma_pos, gamma_neg=gamma_neg, margin=margin, ls=ls, penalty_grad=penalty_grad)
    elif mode == 'dynamic':
        return ASL_dynamic(gamma_pos=gamma_pos, gamma_neg=gamma_neg, margin=margin, dp_target=dp_target, ss=ss, ls=ls, gamma_neg_min=gamma_neg_min, penalty_grad=penalty_grad)
    return

class BCEWithLogitsLoss():
    def __init__(self):
        self.loss = nn.functional.binary_cross_entropy_with_logits
        self.ppm = self.pnm = None
        
    def update(self, pre, target):
        pre = pre.detach()
        target = target.detach()
        ppm  = (pre*target).sum()/target.sum()
        pnm = ((1-pre)*(1-target)).sum()/(1-target).sum()
        
        self.ppm = ppm.item()
        self.pnm = pnm.item()
        return
        
    def __call__(self, pre:torch.Tensor, gt:torch.Tensor, reduction='mean'):
        gt = gt.to(torch.float)
        with torch.no_grad():
            self.update(pre, target=gt)
        return self.loss(pre, gt, reduction=reduction)


class ASL_normal():
    def __init__(
        self,
        gamma_pos = 0.,
        gamma_neg = 4.,
        margin = 0.05,
        eps = 1e-8,
        ls = 0.,
        penalty_grad = False,
    ):
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.margin = margin
        self.eps = eps
        self.ppm = self.pnm = 0.
        self.penalty_grad = penalty_grad
        
    def update(self, pre, target):
        pre = pre.detach()
        target = target.detach()
        ppm  = (pre*target).sum()/target.sum()
        pnm = ((1-pre)*(1-target)).sum()/(1-target).sum()
        
        self.ppm = ppm.item()
        self.pnm = pnm.item()
        return


    def __call__(self, pre:torch.Tensor, gt:torch.Tensor, reduction='mean'):
        p = pre
        pm = torch.clip(p-self.margin, min=0)
        
        pt = p*gt + (1-pm)*(1-gt)
            
        with torch.no_grad():
            self.update(p, gt)
            
        gamma = self.gamma_pos*gt + self.gamma_neg*(1-gt)
        penalty = torch.pow((1-pt), gamma)
        ce = -torch.log(pt.clip(min=self.eps))
        
        if not self.penalty_grad or self.gamma_neg < 1.:
            penalty = penalty.detach()
            
        loss = penalty*ce
        
        if reduction == "none":
            pass
        elif reduction == "mean":
            loss = loss.mean()
        elif reduction == "sum":
            loss = loss.sum()
        else:
            raise ValueError(
                f"Invalid Value for arg 'reduction': '{reduction} \n Supported reduction modes: 'none', 'mean', 'sum'"
        )
        
        return loss
    
class ASL_dynamic(ASL_normal):
    def __init__(
            self, 
            gamma_pos=0, 
            gamma_neg=4, 
            margin=0.05, 
            dp_target=0.1, 
            ss=0.01, 
            ls=0., 
            gamma_neg_min=1.,
            penalty_grad = False,
            ):
        super().__init__(gamma_pos, gamma_neg, margin, ls=ls, penalty_grad=penalty_grad)
        self.dp_target = dp_target
        self.step_size = ss
        self.ppm = self.pnm = 0.
        self.gamma_neg_min = gamma_neg_min
                
    def update(self, pre, target):
        pre = pre.detach()
        if len(pre.shape) == 3:
            pre = pre.mean(dim=1)
            target = target[:,0,:]
            
        target = target.detach()
        if target.sum() == 0:
            ppm = target.sum()
        else:
            ppm  = (pre*target).sum()/target.sum()
        pnm = ((1-pre)*(1-target)).sum()/(1-target).sum()
        
        self.ppm = ppm.item()
        self.pnm = pnm.item()
        
        dp = self.ppm - self.pnm
        
        if self.ppm < 0.98:
            temp_dp_target = ((1-self.ppm)/(1-0.7))*self.dp_target
            temp_dp_target = max(min(temp_dp_target, self.dp_target), 0)
        else:
            temp_dp_target = 0
        # print()
        self.gamma_neg = self.gamma_neg - self.step_size*(dp-temp_dp_target)

        self.gamma_neg = max(self.gamma_neg, self.gamma_neg_min)
        return
    
    