from torch import nn
import torch
import models.classifier as classifier
from torch.nn.utils.parametrizations import spectral_norm
import os
import math
import models.resnet as resnet
from loss.ASL import ASL

def apply_spectral_norm_to_conv(module):
    if isinstance(module, torch.nn.Conv2d):
        return spectral_norm(module)
    
    for child_name, child_module in module.named_children():
        module.add_module(child_name, apply_spectral_norm_to_conv(child_module))
    
    return module

def get_fe(fe_module, in_channels, pre_train):
    if 'resnet' in fe_module:
        model_depth = int(fe_module.replace('resnet',''))
        fe = resnet.resnet_FE(model_depth, in_channel=in_channels, pre_train=pre_train)
    else:
        raise ValueError(f'Current version only support Resnet FE.')
    
    return fe

def get_loss_function(model, args):
    if args.loss_fn == 'bce':
        model.asl_loss = ASL(mode='normal', gamma_pos=0., gamma_neg=0., margin=0.)
        model.loss_fn = model.asl_loss
    elif 'asl' in args.loss_fn: 
        lf = args.loss_fn.replace('asl', '').split('__')
        mode = lf[0]
        asl_args = {
            'gamma_pos': 0.,
            'gamma_neg': 4.,
            'margin': 0.0,  
            'mode': 'dynamic',
            'dp_target': 0.2,
            'ss': 0.001,
            'ls': 0.,
            'gamma_neg_min': 0.,
            'penalty_grad': False,
        }
        for a in lf[1:]:
            k = a[:2]
            v = a[2:]
            if k == 'gp':
                asl_args['gamma_pos'] = float(v)
            elif k == 'gn':
                asl_args['gamma_neg'] = float(v)
            elif k == 'ma':
                asl_args['margin'] = float(v)/100.
            elif k == 'dt':
                asl_args['dp_target'] = float(v)/100.
            elif k == 'ss':
                asl_args['ss'] = float(v.replace('_', '.'))
            elif k == 'ls':
                asl_args['ls'] = float(v.replace('_', '.'))
            elif k == 'nm':
                asl_args['gamma_neg_min'] = float(v.replace('_', '.'))
            elif k == 'pg':
                asl_args['penalty_grad'] = bool(int(v))
            else:
                print(k)
                raise ValueError('look over here')
            
        model.asl_loss = ASL(**asl_args)
        model.loss_fn = model.asl_loss

        
    

class OOD_model(nn.Module):
    def __init__(self, in_channels, n_classes, pre_train=False, fe_module='resnet50', cla_module='at2048'):
        super().__init__()
        self.n_classes = n_classes
        self.fe = get_fe(fe_module=fe_module, in_channels=in_channels, pre_train=pre_train)
        
        if 'at_' in cla_module:
            decoding = cla_module.split('__')
            at_module_index = decoding[0].replace('at_', '')
            latent_size = -1
            n_latent = 2048
            n_layers = 1
            for arg in decoding[1:]:
                if 'ls' in arg:
                    latent_size = int(arg.replace('ls', ''))
                elif 'nl' in arg:
                    n_latent = int(arg.replace('nl', ''))
                elif 'de' in arg:
                    n_layers = int(arg.replace('de', ''))
            c_module = getattr(classifier, f'attention_classifier_{at_module_index}')
            self.cla = c_module(in_channels=self.fe.out_channel, 
                                n_classes=n_classes, 
                                latent_size=latent_size, 
                                n_latent = n_latent,
                                n_layers=n_layers,
                                output_type='prob')
        elif 'JEL' in cla_module:
            decoding = cla_module.split('__')
            n_layers = 2
            if len(decoding)>1:
                for arg in decoding:
                    if 'de' in arg:
                        n_layers = int(arg.replace('de', ''))
            self.cla = classifier.JEL_classifier(self.fe.out_channel, n_classes=n_classes, n_layers=n_layers, output_type='logit')
        else:
            raise ValueError(f'Not suppot cla_module {cla_module}')
        

    def forward(self, x):
        x = self.fe(x)
        x = self.cla(x)
        
        return x
    
    def get_E_yi_given_x_from_output(self, outpt):
        prob = self.cla.get_prob_from_output(outpt)
        out = -(prob).clamp(min=1e-20, max=1).log()
        return out
        
    def get_E_yi_bar_given_x_from_output(self, outpt):
        out = self.cla.get_E_yi_bar_given_x_from_output(outpt)
        return out
    
    def get_Ex_from_output(self, outpt):
        return self.cla.get_Ex_from_output(outpt)
    
    def get_prob_from_output(self, outpt):
        return self.cla.get_prob_from_output(outpt)
    
    def get_Exy2_from_output(self, outpt):
        return self.cla.get_Eyx_from_output(outpt)
        
    
    
    
def get_model(args, device):
    
    model = OOD_model(in_channels=args.in_channels,
                      n_classes=args.n_classes,
                      pre_train=args.pre_train,
                      fe_module=args.fe_module,
                      cla_module=args.cla_module,
                      )
    
    if args.load_path is not None:
        print(f"loading model from {args.load_path}")
        ckpt_dict = torch.load(args.load_path, weights_only=False)
        model.load_state_dict(ckpt_dict["model_state_dict"], strict=False)
        
    get_loss_function(model, args)
        
    if 'SNoJoE' in args.cla_module:
        if args.use_spectral_norm:
            model.fe = apply_spectral_norm_to_conv(model.fe)
    
    return model

def get_model_fe_only(args, device):
    fe_module = args.fe_module
    in_channels = args.in_channels
    pre_train = args.pre_train
    model = get_fe(fe_module=fe_module, in_channels=in_channels, pre_train=pre_train)
    if args.load_path is not None:
        print(f"loading model from {args.load_path}")
        ckpt_dict = torch.load(args.load_path, weights_only=False)
        model.fe.load_state_dict(ckpt_dict["model_state_dict"], strict=True)
    return model

def get_model_fe_pre_train(args, device):
    
    if 'random' in args.load_cluster:
        latent_size = int(args.load_cluster.replace('random', ''))
    else:
        if (args.load_group == 'val') or (args.load_group == 'train'):
            fv_list = torch.load(args.fe_pretrain_path + f'/fv_list_{args.load_group}.pt')
            latent_group = torch.load(args.fe_pretrain_path + f'/latent_group_{args.load_group}_{args.load_cluster}.pt')
            
        n_group = latent_group.amax().item() + 1
        latent_size = int(n_group * 1.)  
        
    args.cla_module = f'{args.cla_module_pre}__ls{latent_size}'
    
    model = OOD_model(in_channels=args.in_channels,
                      n_classes=args.n_classes,
                      pre_train=args.pre_train,
                      fe_module=args.fe_module,
                      cla_module=args.cla_module,
                      )
    
    if args.load_path is not None:
        1/0
    
    if 'random' in args.load_cluster:
        if hasattr(model.cla, 'prototype'):
            prototype_init = model.cla.prototype.data
            with torch.no_grad():
                for i in range(latent_size):
                    temp = prototype_init[:,i]
                    prototype_init[:,i] = temp/torch.norm(temp)
                model.cla.prototype.data = prototype_init
        if args.fe_pretrain_path is not None:
            print('Loading pretrain fe')
            fe_weight_path = args.fe_pretrain_path + '/fe.pt'
            if os.path.exists(fe_weight_path):
                ckpt_dict = torch.load(fe_weight_path, weights_only=False)
                model.fe.fe.load_state_dict(ckpt_dict["model_state_dict"], strict=True)
            else:
                print('Fe path not found. Using imagenet 1k pretrain weight')
    else:
        print('Loading pretrain fe')
        fe_weight_path = args.fe_pretrain_path + '/fe.pt'
        if os.path.exists(fe_weight_path):
            ckpt_dict = torch.load(fe_weight_path, weights_only=False)
            model.fe.fe.load_state_dict(ckpt_dict["model_state_dict"], strict=True)
        else:
            print('Fe path not found. Using imagenet 1k pretrain weight')
            
        
        prototype_init = torch.randn(
            (model.fe.out_channel, latent_size)
            )
        nn.init.kaiming_uniform_(prototype_init, a=math.sqrt(5))
        for i in range(n_group):
            tg = (latent_group == i)
            if tg.sum() == 0:
                continue
            t_fv = fv_list[tg,:]
            t_fv_mean = (t_fv / torch.norm(t_fv, dim=1, keepdim=True)).mean(dim=0)
            t_fv_norm = t_fv_mean / torch.norm(t_fv_mean)
            prototype_init[:,i] = t_fv_norm
        for i in range(n_group, latent_size):
            temp = prototype_init[:,i]
            prototype_init[:,i] = temp/torch.norm(temp)
            
        with torch.no_grad():
            model.cla.prototype.data = prototype_init

    get_loss_function(model, args)

    return model

if __name__ == '__main__':
    # testing
    pass
    
    in_channel = 3
    inpt_size = 256
    
    for cm in [
        'at_u5__ls20__de1',
        'at_u5__ls80__de1',
        'JEL',
      
    ]:
        model = OOD_model(in_channel, 6, fe_module='resnet101', cla_module=cm)
        model.to('cuda')
        x = torch.randn((10, in_channel, inpt_size, inpt_size), dtype=torch.float).to('cuda')
        y = model(x)
        print(y.shape)
        