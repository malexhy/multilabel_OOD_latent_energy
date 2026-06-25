from torch import nn
import torch
from .model_helper import *


    
class JE_classifier(nn.Module):
    def __init__(self, in_channels, n_classes, n_layers=1, output_type='prob') -> None:
        super().__init__()
        self.n_classes = n_classes
        self.output_type = output_type
        self.dummy_Ex = nn.Parameter(torch.zeros(1,1), requires_grad=False)
        
        if output_type == 'prob':
            last_act = nn.Sigmoid
        elif output_type == 'logit':
            last_act = nn.Identity
            
        self.ca = nn.Sequential(
            nn.Linear(in_channels, 1000, bias=True),
            nn.BatchNorm1d(1000),
            nn.ReLU(inplace=True),
            nn.Linear(1000, n_classes,bias=True),
            last_act(),
        )
        
        
    def forward(self, x):
        b,*_ = x.shape
        dummy_Ex = self.dummy_Ex.expand(b, 1)
        return torch.cat((self.ca(x), dummy_Ex), dim=-1)
    
    def get_prob_from_output(self, outpt):
        if self.output_type == 'prob':
            return outpt[:,:-1]
        elif self.output_type == 'logit':
            return outpt[:,:-1].sigmoid()
        
    def get_Ex_from_output(self, outpt):
        return outpt[:,-1]

    def get_Eyx_from_output(self, outpt):
        if self.output_type == 'logit':
            p = outpt[:,:-1]
            return -p
        else:
            raise TypeError('Please use "logit" as output type only.')
    
    def get_E_yi_bar_given_x_from_output(self, outpt):
        if self.output_type == 'prob':
            prob = self.cla.get_prob_from_output(outpt)
            out = -(1-prob).clamp(min=1e-20, max=1).log()
        elif self.output_type == 'logit':
            out = outpt.exp().add(1).log()
        return out
    
    def change2logit(self):
        if isinstance(self.ca[-1], nn.Sigmoid):
            self.ca[-1] = nn.Identity()
        self.output_type = 'logit'

class JEL_classifier(JE_classifier):
    def __init__(self, in_channels, n_classes, n_layers=1, output_type='prob'):
        super().__init__(in_channels, n_classes, n_layers, output_type)
        
        if output_type == 'prob':
            last_act = nn.Sigmoid
        elif output_type == 'logit':
            last_act = nn.Identity
        
        ca = [
            nn.Linear(in_channels, 1000, bias=True),
            nn.ReLU(inplace=True),
        ]
        for i in range(n_layers):
            ca.append(nn.Linear(1000, 1000, bias=True))
            ca.append(nn.ReLU(inplace=True))
            
        ca.append(nn.Linear(1000, n_classes, bias=True))
        ca.append(last_act())
        self.ca = nn.Sequential(*ca)

    
    
class attention_classifier_u(nn.Module):
    def __init__(self, in_channels, n_classes, n_layers=-1, latent_size=100, n_latent=None, output_type='prob', need_Ex_train=False):
        super().__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.latent_size = latent_size
        self.output_type = output_type
        self.need_Ex_train = need_Ex_train
        
        self.dummy_Ex = nn.Parameter(torch.zeros(1,1), requires_grad=False)
        self.prototype = nn.Parameter(
            torch.randn(in_channels, latent_size),
            requires_grad=True
        )
        self.mask_mlp = nn.Sequential(
            up_scale_n_linear(in_channels, 1000, repeat_n=latent_size),
            nn.ReLU(),
            ind_n_linear(1000, n_classes, repeat_n=latent_size),
            nn.Sigmoid()
        )
        
        self.get_Ex = lambda x: -torch.logsumexp(x, dim=-1).unsqueeze(dim=-1)
        self.return_latent_p = False
        self.p_transform = nn.Identity()
    
    def prototype_projection(self, x):
        out = torch.einsum('bf,fl->bl', x, self.prototype)
        return out
        
    def forward(self, x):
        n_E_l_and_x = self.p_transform(self.prototype_projection(x))
        p_l_given_x = (n_E_l_and_x).softmax(dim=-1).unsqueeze(dim=1)
        p_y_given_l_and_x = self.mask_mlp(x)
        
        p_y_given_x = (p_l_given_x @ p_y_given_l_and_x).squeeze(dim=1)   # b, nc
        
        if not self.training or self.need_Ex_train:
            E_x = self.get_Ex(n_E_l_and_x)                 # b, 1

        else:
            b = x.shape[0]
            E_x = self.dummy_Ex.expand(b, 1)
        
        outpt = torch.cat((p_y_given_x, E_x), dim=-1)
        if self.return_latent_p:
            return p_y_given_l_and_x, p_l_given_x, outpt
        else:
            return outpt
        
    def get_prob_from_output(self, outpt):
        return outpt[...,:-1]

    def get_Ex_from_output(self, outpt):
        return outpt[...,-1]
    
    def get_Eyx_from_output(self, outpt):
        prob = self.get_prob_from_output(outpt)  # b, nc
        Ex = self.get_Ex_from_output(outpt)[...,None]    # b, 1
        Ex = Ex.clip(min=-88)
        Eyx = (prob*torch.exp(-Ex)).log().neg()
        return Eyx

    def get_E_yi_bar_given_x_from_output(self, outpt):
        prob = self.get_prob_from_output(outpt)
        out = -(1-prob).clamp(min=1e-20, max=1).log()
        return out
            
class attention_classifier_u5(attention_classifier_u):
    def __init__(self, in_channels, n_classes, n_layers=3, latent_size=100, n_latent=None, output_type='prob', need_Ex_train=False):
        super().__init__(in_channels, n_classes, n_layers, latent_size, n_latent, output_type, need_Ex_train)
        mask_mlp = [
            up_scale_n_linear(in_channels, 128, repeat_n=latent_size),
            nn.ReLU(inplace=True),
        ]
        for i in range(n_layers):
            mask_mlp += [
                ind_n_linear(128, 128, repeat_n=latent_size),
                nn.ReLU(inplace=True),
            ]
        
        mask_mlp += [
            ind_n_linear(128, n_classes, repeat_n=latent_size),
            nn.Sigmoid(),
        ]
            
        self.mask_mlp = nn.Sequential(
            *mask_mlp
        )
        
class attention_classifier_u5MoE2(attention_classifier_u):
    def __init__(self, in_channels, n_classes, n_layers=-1, latent_size=100, n_latent=None, output_type='prob', need_Ex_train=False):
        super().__init__(in_channels, n_classes, n_layers, latent_size, n_latent, output_type, need_Ex_train)
    
    def forward(self, x):
        n_E_l_and_x = self.p_transform(self.prototype_projection(x)) # b, L
        top1_logits, top1_indices = torch.topk(n_E_l_and_x, k=2, dim=-1) # b, k
        
        
        p_l_given_x = (top1_logits).softmax(dim=-1) # b, k
        
        p_y_given_l_and_x = self.mask_mlp(x) # b, L, nc
        p_y_given_l_and_x = torch.gather(p_y_given_l_and_x, dim=1, index=top1_indices.unsqueeze(-1).expand(-1, -1, self.n_classes))
        p_l_given_x = p_l_given_x.unsqueeze(dim=1) 
        
        p_y_given_x = (p_l_given_x @ p_y_given_l_and_x).squeeze(dim=1)   # b, nc
        
        if not self.training or self.need_Ex_train:
            E_x = self.get_Ex(n_E_l_and_x)                 # b, 1

        else:
            b = x.shape[0]
            E_x = self.dummy_Ex.expand(b, 1)
        
        outpt = torch.cat((p_y_given_x, E_x), dim=-1)
        if self.return_latent_p:
            return p_y_given_l_and_x, p_l_given_x, outpt
        else:
            return outpt
        

class attention_classifier_u6(attention_classifier_u5):
    def __init__(self, in_channels, n_classes, n_layers=3, latent_size=100, n_latent=None, output_type='prob', need_Ex_train=False):
        super().__init__(in_channels, n_classes, n_layers, latent_size, n_latent, output_type, need_Ex_train)
        if isinstance(self.mask_mlp[-1], nn.Sigmoid):
            self.mask_mlp[-1] = nn.Identity()
    
    def forward(self, x):
        n_E_l_and_x = self.p_transform(self.prototype_projection(x)).unsqueeze(dim=-1) # b, nl, 1
        belta = self.mask_mlp(x)    #b, nl, nc
        alpha = nn.functional.softplus(-belta)  #b, nl, nc
        
        Elx = -n_E_l_and_x  # b, nl, 1
        Eyxl = Elx + alpha  #b, nl, nc
        
        n_Ex = torch.logsumexp(n_E_l_and_x, dim=1) # b, 1
        n_Eyx = torch.logsumexp(-Eyxl, dim=1) # b, nc
        
        prob = torch.exp(n_Eyx-n_Ex)
        
        outpt = {
            'prob': prob,
            'Ex': -n_Ex,
            'Eyx': -n_Eyx,
        }
        if self.return_latent_p:
            p_l_given_x = (n_E_l_and_x).softmax(dim=-1).unsqueeze(dim=1) 
            p_y_given_l_and_x = belta.sigmoid()
            return p_y_given_l_and_x, p_l_given_x, outpt
        else:
            return outpt
        
    def get_prob_from_output(self, outpt):
        return outpt['prob']
    
    def get_Ex_from_output(self, outpt):
        return outpt['Ex'].squeeze(dim=-1)
    
    def get_Eyx_from_output(self, outpt):
        return outpt['Eyx']
    
    def get_E_yi_bar_given_x_from_output(self, outpt):
        prob = self.get_prob_from_output(outpt)
        out = -(1-prob).clamp(min=1e-20, max=1).log()
        return out