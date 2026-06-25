from train_base import train
import argparse
from copy import deepcopy
import os
from cluster import cluster
import utils
import json

def main(args):
    
    os.makedirs(args.save_dir, exist_ok=True)
    if args.train_data_ind is not None:
        args.train_data_ind_path = os.path.join('./dataset', 'data_ind', args.train_data_ind)
    else:
        args.train_data_ind_path = None
    
    train_set = args.training_set
    eval_every = args.eval_every
    n_epochs = args.n_epochs
    if '_' in train_set:
        if 'voc' in train_set:
            pres = int(train_set.split('_')[-1])/100
            m_eval_every = int(eval_every/pres)
            m_n_epochs = int(n_epochs/pres)
        elif 'nus' in train_set:
            p, n = utils.nus_decode(train_set)
            if (p is not None) and (n is not None):
                print('Cannot use both')
                1/0
                        
            if n is not None:
                p = max(n/119986, 0.1)
                
            m_eval_every = int((eval_every/p)/2)
            m_n_epochs = int(n_epochs/p)
    else:
        m_eval_every = eval_every
        m_n_epochs = n_epochs
        
        
    if 'at' in args.cla_module:
        
        if args.from_cluster is None and args.from_pre_train is None:
            pre_train_save_path = os.path.join(args.save_dir, args.pretrain_dir_name)
            args_temp = deepcopy(args)
            print('Pretraining model ......')
            args_temp.save_dir = pre_train_save_path
            args_temp.fe_lr_rate = 1.0
            args_temp.batch_size = 128
            args_temp.lr = 1e-4
            args_temp.eval_every = m_eval_every
            args_temp.n_epochs = m_n_epochs
            if args.quick_mode:
                if args.training_set == 'voc':
                    args_temp.use_ema = 0
            if args.ID_pretrain:
                pass
                # if os.path.exists(pre_train_save_path):
                #     print('print trained weight exsit, skiping pre_train')
                # else:
                train(args_temp, stage='pretrain')
            else:
                train(args_temp, stage='dummy')
            
            print('Finished pretrain model!')
            del args_temp
            
        else:
            if args.from_cluster is not None :
                if not os.path.exists(args.from_cluster):
                    print('Cluster path not found!')
                    1/0
                pre_train_save_path = os.path.dirname(args.from_cluster)
            elif args.from_pre_train is not None :
                if not os.path.exists(args.from_pre_train):
                    print('Cluster path not found!')
                    1/0
                pre_train_save_path = args.from_pre_train
                
            with open(os.path.join(pre_train_save_path, 'params.txt'), 'r') as f:
                log_dict = json.load(f)
            log_args = argparse.Namespace(**log_dict)
            args.training_set = log_args.training_set
            args.eval_every = log_args.eval_every
            args.n_epochs = log_args.n_epochs
            args.fe_module = log_args.fe_module
            args.train_data_ind = log_args.train_data_ind
            args.im_size = log_args.im_size
                
        
        if args.from_cluster is None:
            if 'kmeans' in args.cluster_method:
                print('Clustering ......')
                if args.ID_pretrain:
                    best = utils.find_best_result(pre_train_save_path, 'mAP')
                    print(f'Best mAP ckpt = {best}')
                    utils.remove_non_best(pre_train_save_path, best, True)
                    ckpt = best
                else:
                    ckpt = 'none'
                    os.makedirs(os.path.join(pre_train_save_path, ckpt), exist_ok=True)
                cluster(
                    load_path=pre_train_save_path,
                    ckpt = ckpt,
                    save_group=True,
                    save_figure=False,
                    sub_version=args.cluster_sub_version,
                    cluster_method=deepcopy(args.cluster_method),
                    use_split=deepcopy(args.load_group),
                    )
            else:
                print('Using random init. No need for cluster.')
        else:
            ckpt = os.path.basename(args.from_cluster)
            
        print('Fine-tuning model')
        args_temp = deepcopy(args)
        args_temp.fe_pretrain_path = os.path.join(pre_train_save_path, ckpt)
        args_temp.save_dir = os.path.join(args.save_dir, args.final_dir_name)
        args_temp.load_cluster = deepcopy(args.cluster_method)
        args_temp.cla_module_pre = deepcopy(args.cla_module)
        args_temp.eval_every = m_eval_every
        args_temp.n_epochs = m_n_epochs
        train(args_temp, stage='final')
    else:
        print('Training model ......')
        
        args_temp = deepcopy(args)
        args_temp.save_dir = os.path.join(args.save_dir, args.final_dir_name)
        args_temp.eval_every = m_eval_every
        args_temp.n_epochs = m_n_epochs
        train(args_temp, stage='final')



if __name__ == '__main__':
    parser = argparse.ArgumentParser("EMB LR")
    parser.add_argument("--in_channels", type=int, default=3)
    parser.add_argument("--n_classes", type=int, default=80)
    parser.add_argument("--fe_module", type=str, default='resnet101', choices=['resnet18','resnet34','resnet50','resnet101'])
    parser.add_argument("--cla_module", type=str, default='at_u5__de1')
    parser.add_argument("--pre_train", type=int, default=1, choices=[0, 1])
    
    #ema model
    parser.add_argument('--use_ema', default=1, type=int, choices=[0, 1],
                        help='use model ema')
    parser.add_argument('--ema_decay', default=0.9997, type=float, metavar='M',
                        help='decay of model ema')
    
    #dataset
    parser.add_argument("--training_set", type=str, default='coco', help="Available: ['coco', 'voc', 'nus', 'voc_*', 'nus_500n', 'nus_10p']")
    parser.add_argument("--train_data_ind", type=str, default=None, choices=['train_ind0', 'train_ind1', 'train_ind2', None, 'train_ind_4_0', 'train_ind_4_2'])
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--im_size", type=int, default=256)
    
    #training
    parser.add_argument("--optimizer", choices=["adam"], default="adam")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.)
    parser.add_argument("--fe_lr_rate", type=float, default=0.1, help='Ratio of learning rate of the feature extractor.')
    parser.add_argument("--pro_lr_rate", type=float, default=None, help='Ratio of learning rate of the prototype. None meaning it uses the same ratio of the feature extractor')
    parser.add_argument("--n_epochs", type=int, default=200)
    parser.add_argument('--loss_fn', type=str, default='asld__dt20__ss0_001__ma0', choices=['asld__dt20__ss0_001__ma0', 'bce'])
    parser.add_argument('--early_stop', default=1, type=int, choices=[0, 1], help='Early stop the model when mAP does not improve.')
    parser.add_argument('--use_tensorboard', default=0, type=int, choices=[0, 1], help='use tensorboard?')
    parser.add_argument('--quick_mode', default=0, type=int, choices=[0, 1], help='Validation took time. Enable this would perform less validation for speeding up.')
    
    
    #logging + evaluation
    parser.add_argument("--eval_every", type=int, default=1, help="Epochs between evaluation")
    parser.add_argument("--print_every", type=int, default=100, help="Iterations between print")
    parser.add_argument("--save_dir", type=str, default='./save/weight_1')
    parser.add_argument("--load_path", type=str, default=None)
    
    #others
    parser.add_argument("--seed", type=int, default=None)
    
    # init_method
    parser.add_argument("--load_group", type=str, default='train', choices=['train', 'val'], help='Using which split of the dataset for clustering, only allow "train" or "val"')
    parser.add_argument("--cluster_method", type=str, default='kmeans_auto', choices=['kmeans_*', 'random*'], help='Using which method to cluster the feature vectors. "kmeans_auto" uses auto clustering, \
        "kmeans_N" uses manually selected N cluster. "randomN" uses random intilization with N latent size.')
    parser.add_argument('--ID_pretrain', default=1, type=int, choices=[0, 1], help='Whether pretraining the feature extractor with in-distribution data.')
    
    parser.add_argument('--pretrain_dir_name', default='pretrain', type=str, help='Name of the sub fodlder to save the pretrain weight.')
    parser.add_argument('--final_dir_name', default='final', type=str, help='Name of the sub fodlder to save the final weight.')
    
    parser.add_argument('--from_pre_train', default=None, type=str, help='Pretrained path')
    parser.add_argument('--from_cluster', default=None, type=str, help='Cluster path')
    parser.add_argument('--cluster_sub_version', default=None, type=str, help='Cluster sub version')
    
    args = parser.parse_args()
    
    main(deepcopy(args))