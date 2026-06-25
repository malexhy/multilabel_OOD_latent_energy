import argparse
import torch
from OOD_model import get_model_fe_pre_train, get_model
import json
import utils 
from dataset.dataset import get_dataLoaders
import random
from torch.amp import GradScaler, autocast
from tqdm import tqdm
from torcheval.metrics.functional import multilabel_auprc as mAP
from torcheval.metrics.functional import multilabel_accuracy as Acc
import os
from models.ema_model import EMA_Model
import numpy as np

def checkpoint_final(model, tag, args, device, log_dict):
    model.cpu()
    ckpt_dict = {
        "model_state_dict": model.state_dict(),
    }
    torch.save(ckpt_dict, os.path.join(args.save_dir, tag))
    save_record_dir = os.path.join(args.save_dir, 'save_log.json')
    if os.path.exists(save_record_dir):
        with open(save_record_dir, 'r') as f:
            sr = json.load(f)
    else:
        sr = {}
    sr[tag] = f'Epoch: {log_dict["epoch"]}, '
    if "tpr" in log_dict.keys():
        sr[tag] += f'TPR: {log_dict["tpr"]:<.5f}, '
    sr[tag] += f'mAP: {log_dict["mAP"]:<.5f}'
    with open(save_record_dir, 'w') as f:
        json.dump(sr, f, indent=3)
        
    model.to(device)

def checkpoint_pretrain(model, tag, args, device, log_dict):
    model.cpu()
    tag = tag.replace('.pt', '')
    ckpt_dict = {
        "model_state_dict": model.fe.fe.state_dict(),
    }
    weight_path = os.path.join(args.save_dir, tag)
    if not os.path.exists(weight_path):
        os.mkdir(weight_path)
    torch.save(ckpt_dict, os.path.join(weight_path, 'fe.pt'))
    # torch.save(full_ckpt_dict, os.path.join(weight_path, 'full.pt'))
    save_record_dir = os.path.join(args.save_dir, 'save_log.json')
    if os.path.exists(save_record_dir):
        with open(save_record_dir, 'r') as f:
            sr = json.load(f)
    else:
        sr = {}
    sr[tag] = f'Epoch: {log_dict["epoch"]}, mAP: {log_dict["mAP"]:<.5f}'
    with open(save_record_dir, 'w') as f:
        json.dump(sr, f, indent=3)
        
    model.to(device)
    
def get_metrics(prob, target, n_classes):
    _mAP = mAP(prob, target, num_labels= n_classes).item()
    acc = Acc(prob, target, criteria="hamming").item()
    tpr = utils.TPR(prob, target, output_fpr=False)
    tpr = tpr.item()
    return _mAP, acc, tpr
        
def eval_classification(model, test_loader, device, args):
    probs = []
    gts = []
    for x_in, y_in in test_loader:
        x_in, y_in = x_in.to(device), y_in.to(device)
        outpt = model(x_in)
        prob = model.get_prob_from_output(outpt)
        probs.append(prob)
        gts.append(y_in)
    probs = torch.cat(probs, dim=0)
    gts = torch.cat(gts, dim=0)
    _mAP, acc, tpr = get_metrics(probs, gts, n_classes=args.n_classes)
    loss = model.loss_fn(probs, gts).item()
    return loss, _mAP, acc, tpr

def get_optim(args, params):
    if args.optimizer == "adam":
        optim = torch.optim.Adam(params, lr=args.lr, betas=[.9, .999], weight_decay=0)
            
    return optim

def get_params(model, weight_decay=0., lr=2e-4, skip_list=(), fe_lr_rate=.1, pro_lr_rate=None):
    if pro_lr_rate is None:
        pro_lr_rate = fe_lr_rate
    decay_fe = []
    no_decay_fe = []
    decay_cla = []
    no_decay_cla = []
    prototype = []
    for name, param in model.fe.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            no_decay_fe.append(param)
        else:
            decay_fe.append(param)
    
    for name, param in model.cla.named_parameters():
        
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            no_decay_cla.append(param)
        elif name == 'prototype':
            prototype.append(param)
        else:
            decay_cla.append(param)

    params = []
    
    params += [
        {'params': no_decay_cla, 'weight_decay': 0., 'lr':lr, 'tag':'cla_no_decay'},
        {'params': decay_cla, 'weight_decay': weight_decay, 'lr':lr, 'tag':'cla_decay'},
    ]
    
    if fe_lr_rate == 0.:
        for param in decay_fe:
            param.requires_grad = False
        for param in no_decay_fe:
            param.requires_grad = False
    else:
        params += [
            {'params': no_decay_fe, 'weight_decay': 0., 'lr':lr*fe_lr_rate, 'tag':'fe_no_decay'},
            {'params': decay_fe, 'weight_decay': weight_decay, 'lr':lr*fe_lr_rate, 'tag':'fe_decay'},
        ]
        
    if pro_lr_rate == 0.:
        for param in prototype:
            param.requires_grad = False
    else:
        params += [
            {'params': prototype, 'weight_decay': weight_decay, 'lr':lr*pro_lr_rate, 'tag':'prototype'},
        ]
    
    return params

def compute_entropy_loss(plx):
    # print(plx.shape)
    if len(plx.shape) == 3:
        plx = plx.flatten(dim=1)
    b, e = plx.shape
    imp = plx.mean(dim=0) # e
    _, topk_indices = torch.topk(plx, k=1, dim=-1)
    one_hot = torch.nn.functional.one_hot(topk_indices.squeeze(-1), num_classes=e).float()
    load = one_hot.mean(dim=0) # e
    loss = e * torch.sum(imp * load)
    return loss

def train(args, stage='final'):
    os.makedirs(args.save_dir, exist_ok=True)
    
    
    if args.seed is None:
        seed = random.randint(0, 1e8)
        args.seed = seed
    else:
        seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
        
    train_loader, test_loader = get_dataLoaders(args)
    args.n_classes = train_loader.dataset.n_classes
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    
    
    
    if stage == 'final':
        if 'at' in args.cla_module and 'MoE' in args.cla_module:
            args.use_entropy = 0.001
        else:
            args.use_entropy = -1

        if 'at' in args.cla_module:
            model = get_model_fe_pre_train(args, device)
        else:
            model = get_model(args, device)
            
        checkpoint = checkpoint_final
            
    elif stage == 'pretrain':
        if 'at' not in args.cla_module:
            print('Pretrain only apply for LAC. Use "cla_module" = "at_u5__de1"')
            1/0
        args.use_entropy = -1
        args.cla_module = 'JEL__de1'
        model = get_model(args, device)
        
        checkpoint = checkpoint_pretrain
        
    elif stage == 'dummy':
        if 'at' not in args.cla_module:
            print('Pretrain only apply for LAC. Use "cla_module" = "at_u5__de1"')
            1/0
        args.use_entropy = -1
        args.cla_module = 'JEL__de1'
        model = get_model(args, device)
        
        checkpoint_pretrain(model, 'none', args, device, {'epoch': 0, 'mAP': 1.})
    else:
        print('Error stage')
        1/0
    
    model.to(device)


    
    
    if getattr(args, 'quick_mode', 0) and (stage == 'final') and ('at' in args.cla_module):
        args.use_tensorboard = False
        if 'coco' in args.training_set:
            args.eval_start_epoch = 10
            args.eval_stop_epoch = None
            args.ema_eval_start_epoch = 15
            
        elif args.training_set == 'voc':
            args.eval_start_epoch = 15
            args.eval_stop_epoch = 50
            args.ema_eval_start_epoch = 110
        else:
            args.eval_start_epoch = -1
            args.eval_stop_epoch = None
            args.ema_eval_start_epoch = -1
    else:
        args.eval_start_epoch = -1
        args.eval_stop_epoch = None
        args.ema_eval_start_epoch = -1
    
    if args.use_ema:
        ema_model = EMA_Model(model=model, decay=args.ema_decay, device=device)
    # if DEBUG:
    #     from torchinfo import summary
    #     summary(model=model, 
    #         input_size=(1, args.in_channels, args.im_size, args.im_size), 
    #         col_names=['input_size', 'output_size', 'num_params'],
    #         depth=3)
    #     return
    
    with open(f'{args.save_dir}/params.txt', 'w') as f:
        json.dump(args.__dict__, f, indent=3)
        
    if stage == 'dummy':
        return

    params = get_params(
        model, 
        args.weight_decay, 
        lr=args.lr,
        fe_lr_rate=args.fe_lr_rate,
        pro_lr_rate=args.pro_lr_rate,
        )

    optim = get_optim(
        args=args, 
        params=params,
        )
    
        
    best_valid_acc = 0.0
    best_valid_acc_ema = 0.0

    eval_stop = getattr(args, 'eval_stop_epoch', None)
    eval_start = getattr(args, 'eval_start_epoch', -1)
    ema_eval_start = getattr(args, 'ema_eval_start_epoch', -1)
    
    
    cur_iter = 0
    
    scaler = GradScaler(device=device)
    print_current_stage = False
    early_stop_count = 0
    
    if args.use_tensorboard:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=f'{args.save_dir}/t_log')
    
    model.train()
    if args.fe_lr_rate == 0.:
        model.fe.eval()
    

    for epoch in range(args.n_epochs):
                                     
        for i, (x_in, y_in) in tqdm(enumerate(train_loader)):
            print_current_stage = (cur_iter % args.print_every == 0)
            print('', end='\r')

            x_in, y_in = x_in.to(device), y_in.to(device)
            with autocast(device_type='cuda'):
                if (args.use_entropy > 0) and ('at' in args.cla_module) :
                    model.cla.return_latent_p = True
                    pylx, plx, outpt = model(x_in)
                    model.cla.return_latent_p = False
                else:
                    outpt = model(x_in)
               
                prob = model.get_prob_from_output(outpt)
                loss = model.loss_fn(prob, y_in)
                if args.use_entropy > 0:
                    # print('adding entropy')
                    loss += args.use_entropy*compute_entropy_loss(plx)
                
            optim.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()

            if args.use_ema:
                ema_model.update(model)
                
                
            cur_iter += 1
            
                
            if print_current_stage:
                loss = loss.detach().item()
                with torch.no_grad():
                    _mAP, acc, tpr = get_metrics(prob, y_in, args.n_classes)
                print(f'Epoch:{epoch:<3d}| i:{i:3d} | Total iter: {cur_iter:<}')
                print(f'   {"P(y|x)":<10}| acc={acc:<8.5f}, TPR={tpr:<8.5f}, mAP={_mAP:<8.5f}')
                print(f'   {"Others":<10}| Total Loss={loss:<8.5f}')
                
                if hasattr(model, 'asl_loss'):
                    if hasattr(model.asl_loss, 'gamma_neg'):
                        print(f'   {" ":<10}| γ={model.asl_loss.gamma_neg:<8.5f}', end=' ')
                    if hasattr(model.asl_loss, 'ppm'):
                        print(f'PP={model.asl_loss.ppm:<8.5f}', end='')
                        print(f'PN={model.asl_loss.pnm:<8.5f}')
                    if hasattr(model.asl_loss, 'gamma_neg_list'):
                        print(f'   {" ":<10}| γ_max={model.asl_loss.gamma_neg_list.amax().item():<8.5f},  γ_min={model.asl_loss.gamma_neg_list.amin().item():<8.5f}')
                        
                
            
        do_eval = ((epoch > eval_start) and ((eval_stop is None) or (eval_stop >= epoch))) 
        do_ema_eval = args.use_ema and (epoch >= ema_eval_start)
        if (epoch % args.eval_every == 0) and (do_eval or do_ema_eval):
            model.eval()
            early_stop_count += 1
            if args.use_tensorboard:
                scalars = {
                    'mAP':{},
                    'TPR':{},
                    'Acc':{},
                }
                if hasattr(model, 'asl_loss'):
                    asl_dict = {
                                '1_gamma-': model.asl_loss.gamma_neg,
                                }
                    writer.add_scalars('ASL',
                                       asl_dict, 
                                       epoch)
    
            with torch.no_grad():
                # validation set
                if do_eval: 
                    loss, _mAP, acc, tpr = eval_classification(model, test_loader, device, args)
                    print(f'{f"Epoch {epoch}: Valid results":#<60}')
                    print(f'   {"P(y|x)":<10}| acc={acc:<8.5f}, TPR={tpr:<8.5f}, mAP={_mAP:<8.5f}')
                    print(f'   {"Others":<10}| Total Loss={loss:<8.5f}')
                    
                    if args.use_tensorboard:
                        scalars['mAP']['Model'] = _mAP
                        scalars['Acc']['Model'] = acc
                        scalars['TPR']['Model'] = tpr
                        
                    if _mAP > best_valid_acc:
                        early_stop_count = 0
                        best_valid_acc =_mAP
                        print("Best Valid Acc!: {}".format(_mAP))
                        checkpoint(model, "best_valid_acc_ckpt.pt", args, device, {'epoch': epoch, 'mAP':_mAP})
                    
                        
                if do_ema_eval:
                    ema_loss, ema_mAP, ema_acc, ema_tpr = eval_classification(ema_model.module, test_loader, device, args)
                    print(f'---EMA{"":-<54}')
                    print(f'   {"P(y|x)":<10}| acc={ema_acc:<8.5f}, TPR={ema_tpr:<8.5f}, mAP={ema_mAP:<8.5f}')
                    print(f'   {"Others":<10}| Total Loss={ema_loss:<8.5f}')
                    
                    if args.use_tensorboard:
                        scalars['mAP']['EMA_Model'] = ema_mAP
                        scalars['Acc']['EMA_Model'] = ema_acc
                        scalars['TPR']['EMA_Model'] = ema_tpr
                    

                    if ema_mAP > best_valid_acc_ema:
                        early_stop_count = 0
                        best_valid_acc_ema = ema_mAP
                        print("Best EMA Valid Acc!: {}".format(ema_mAP))
                        checkpoint(ema_model.module, "best_ema_valid_acc_ckpt.pt", args, device, {'epoch': epoch, 'mAP':ema_mAP})
                        
                print(f'{"":#<60}')

            model.train()
                
        if args.use_tensorboard:
            for k, v in scalars.items():
                writer.add_scalars(k, v, epoch)
        
        if args.early_stop and early_stop_count >= 5:
            if args.use_ema:
                if (epoch >= ema_eval_start):
                    print('EMA performance has not been improved for 5 epoch. Early stop')
                    break
            else:
                print('EMA performance has not been improved for 5 epoch. Early stop')
                break
    
    del model, optim
        
    
    
DEBUG = True

    
if __name__ == "__main__":
    pass
    # from glob import glob
    # parser = argparse.ArgumentParser("EMB LR")
    
    # # model
    # parser.add_argument("--in_channels", type=int, default=3)
    # parser.add_argument("--n_classes", type=int, default=80)
    # parser.add_argument("--fe_module", type=str, default='resnet101', choices=['resnet18','resnet34','resnet50','resnet101'])
    # parser.add_argument("--cla_module", type=str, default='at_u5__de1__ls20', choices=['at*', 'JE', 'BENN', 'SNoJoE'])
    # parser.add_argument("--pre_train", type=int, default=1, choices=[0, 1])
    
    # #ema model
    # parser.add_argument('--use_ema', default=0, type=int, choices=[0, 1],
    #                     help='use model ema')
    # parser.add_argument('--ema_decay', default=0.9997, type=float, metavar='M',
    #                     help='decay of model ema')
    
    # #dataset
    # parser.add_argument("--training_set", type=str, default='voc', choices=['coco', 'voc', 'nus', 'voc_*', 'nus_500n', 'nus_10p'])
    # parser.add_argument("--batch_size", type=int, default=128)
    # parser.add_argument("--im_size", type=int, default=256)
    
    # #training
    # parser.add_argument("--optimizer", choices=["adam"], default="adam")
    # parser.add_argument("--lr", type=float, default=2e-4)
    # parser.add_argument("--weight_decay", type=float, default=0.)
    # parser.add_argument("--n_epochs", type=int, default=200)
    # parser.add_argument('--loss_fn', type=str, default='asld', choices=['asld', 'bce'])
    # parser.add_argument('--early_stop', default=1, type=int, choices=[0, 1], help='Early stop the model when mAP does not improve.')
    # parser.add_argument('--use_tensorboard', default=0, type=int, choices=[0, 1], help='use tensorboard?')
    # parser.add_argument('--quick_mode', default=0, type=int, choices=[0, 1], help='Validation took time. Enable this would perform less validation for speeding up.')
    
    
    # #logging + evaluation
    # parser.add_argument("--eval_every", type=int, default=1, help="Epochs between evaluation")
    # parser.add_argument("--print_every", type=int, default=100, help="Iterations between print")
    # parser.add_argument("--save_dir", type=str, default='./save/weight_1')
    # parser.add_argument("--load_path", type=str, default=None)
    
    # #others
    # parser.add_argument("--seed", type=int, default=None)
    
    # # init_method
    # parser.add_argument("--load_group", type=str, default='train', choices=['train', 'val'], help='Using which split of the dataset for clustering, only allow "train" or "val"')
    # parser.add_argument("--load_cluster", type=str, default='kmeans_auto', choices=['kmeans_*', 'random*'], help='Using which method to cluster the feature vectors. "kmeans_auto" uses auto clustering, \
    #     "kmeans_N" uses manually selected N cluster. "randomN" uses random intilization with N latent size.')
    
    
    
    # args = parser.parse_args()
        
    
    # train(args)
         
