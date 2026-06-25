import torch 
import argparse
import json
from OOD_model import get_model
import glob
import argparse

from test_method.OOD_metric import OOD_test
        

def main(load_path, weight, method_args):
    with open(f'{load_path}/params.txt', 'r') as f:
        log_dict = json.load(f)
    log_args = argparse.Namespace(**log_dict)
    log_args.load_path = load_path+'/'+weight
    
    if method_args['eval_dataset'] == None:
        ts = log_args.training_set
        if 'coco' in ts:
            ed = ['coco_test',]
        elif 'voc' in ts:
            ed = ['voc_test',]
        elif 'nus' in ts:
            ed = ['nus_test',]
        ed += ['DTD', 'imagenet']
        method_args['eval_dataset'] = ed
            
            
    if method_args['inspect_target'] == None:
        ts = log_args.training_set
        if 'coco' in ts:
            it = 'coco_test'
        elif 'voc' in ts:
            it = 'voc_test'
        elif 'nus' in ts:
            it = 'nus_test'
        method_args['inspect_target'] = it

            
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
    model = get_model(log_args, device)

    model.to(device)
    model.eval()

    
    if 'JE' in log_args.cla_module:
        method_args['score_fn']=['Eybarx_sum', 'Eyx2_min']
        model.cla.change2logit()
    elif 'at' in log_args.cla_module:
        method_args['score_fn']=['Eyx2_min', 'Ex']
    elif 'BENN' == log_args.cla_module:
        method_args['score_fn']=['Ex']
        
    if 'SNoJoE' in log_args.cla_module:
        method_args['score_fn']=['Eybarx_sum']
        
    if DEBUG:
        from torchinfo import summary
        summary(
            model=model, 
            input_size=(8,log_args.in_channels,log_args.im_size,log_args.im_size), 
            col_names=['input_size', 'output_size', 'num_params'], 
            depth=3)
        exit()
        
        

    method_args['eval_dataset'] = [d for d in method_args['eval_dataset'] if d[0] != '!']
    
    


        
    eval_dataset = method_args['eval_dataset']
    inspect_target = method_args['inspect_target']
    _score_fn =  method_args['score_fn'].copy()

    _score_fn = [fn for fn in _score_fn if fn[0] != '!']
    if isinstance(_score_fn, list):
        for sf in _score_fn:
            OOD_test(load_path=load_path,
                    weight=weight,
                    eval_dataset=eval_dataset,
                    inspect_target=inspect_target,
                    model=model,
                    device=device,
                    score_fn=sf,
                    args=log_args,
                    save_name='OOD_result',
                    )
    else:
        OOD_test(load_path=load_path,
                weight=weight,
                eval_dataset=eval_dataset,
                model=model,
                device=device,
                score_fn=_score_fn,
                args=log_args,
                save_name='OOD_result')
            
            
if __name__ == "__main__":
    import utils
    import os
    
    ACCEPT_WEIGHT = [
        'best_mAP',
        'best_ema_valid_acc_ckpt.pt',
        'best_valid_acc_ckpt.pt',
    ]
    parser = argparse.ArgumentParser("EMB LR")
    
    parser.add_argument("--eval_path", type=str, default='./save/weight_0', help='Path to eval.')
    parser.add_argument("--remove_non_best", type=int, default=0, choices=[0,1], help='Remove non best weight for saving hard drive memory.')
    parser.add_argument("--weight_path", type=str, default='best_mAP', help='Weight eval.', choices=ACCEPT_WEIGHT, help='Specific .pt weigth to eval. Default "best_mAP" choose the weight with the best val "mAP".')

    args = parser.parse_args()
    

    
    method_args = {
                   'eval_dataset':None,
                   'score_fn':[],
                   'inspect_target':None,
                   }
    
    DEBUG = False
    # paths = [os.path.join(p, 'final') for p in sorted(glob.glob(args.eval_path))]
    paths = sorted(glob.glob(args.eval_path))
    for load_path in paths:
        print(load_path)
        if 'OOD_result_metric' in load_path:
            continue
        
        if args.weight_path == 'best_mAP':
            best = utils.find_best_result(load_path, target='mAP')
            print(best) 
            if args.remove_non_best:
                utils.remove_non_best(load_path, best, confirm=True)
            weight = best
        else:
            weight = args.weight_path


        main(load_path=load_path, weight=weight,  method_args=method_args)
