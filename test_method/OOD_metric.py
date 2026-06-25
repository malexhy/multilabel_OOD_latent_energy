import torch
from torchvision.transforms import v2
import torchvision as tv
import matplotlib.pyplot as plt
from dataset.pascalVOC_dataset2 import pascalVOC
from dataset.nus_wide_dataset import nuswide
from torch.utils.data import DataLoader
import scipy.stats as ss
import numpy as np
import utils

from torcheval.metrics.functional import binary_auroc as auroc
from torcheval.metrics.functional import binary_auprc as auprc
from tqdm import tqdm
import os
from utils import fpr_and_fdr_at_recall


IN_SMALLER_LIST = ['Ex', 'Eyx2_sum', 'Eyx_min', 'combin_sum', 'Eyx2_max', 'Eyx2_min']
IN_LARGER_LIST = ['Eybarx_sum', 'Eyx_max', 'Eyx_sum', 'combin_sum2']

class DTD_SubLoader(tv.datasets.DTD):
    def __init__(self, root, split = "train", partition = 1, transform = None, target_transform = None, download = False):
        super().__init__(root, split, partition, transform, target_transform, download)
        
    def __getitem__(self, idx):
        x, _ = super().__getitem__(idx)
        return x, torch.zeros((1,))

def get_dataset(datasets, inspect_target, args):

    from dataset.coco_dataset2 import coco 
 
        
    normalize = v2.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    if args.in_channels == 1:
        transform_test = v2.Compose([
            v2.Grayscale(),
            v2.Resize(args.im_size),
            v2.ToTensor(),
        ])
        transform_test_coco = v2.Compose([
            v2.Grayscale(),
            v2.CenterCrop((args.im_size, args.im_size)),
            v2.SanitizeBoundingBoxes(10.),
            normalize,
        ])
    elif args.in_channels == 3:
        transform_test = v2.Compose([
            v2.Resize((args.im_size, args.im_size)),
            v2.ToImage(), 
            v2.ToDtype(torch.float32, scale=True),
            normalize,
        ])
        transform_test_coco = v2.Compose([
            v2.Resize((args.im_size, args.im_size)),
            normalize,
        ])
        transform_test_dtd = v2.Compose([
            v2.ToImage(), 
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize((args.im_size, args.im_size)),
            normalize,

        ])
    datasets_dict = {}
    for d in datasets:
        if d == "coco_train":
            datasets_dict[d] = coco(split='train', img_transform=transform_test_coco, included_classes=None)
        elif d == "coco_val":
            datasets_dict[d] = coco(split='val', img_transform=transform_test_coco, included_classes=None)
        elif d == "coco_test":
            datasets_dict[d] = coco(split='test', img_transform=transform_test_coco, included_classes=None)
        elif d == "DTD":
            datasets_dict[d] = DTD_SubLoader(root='../data', transform=transform_test_dtd, split='train')
        elif d == 'imagenet':
            if 'nus' in inspect_target:
                print('Using nus_image_net')
                datasets_dict[d] = tv.datasets.ImageFolder(root='../data/ImageNet_22k_subset_nus_3', transform=transform_test_dtd)
            else:
                print('Using non_nus_image_net')
                datasets_dict[d] = tv.datasets.ImageFolder(root='../data/ImageNet_22k_subset_2', transform=transform_test_dtd)
        elif d == 'voc_test':
            datasets_dict[d] = pascalVOC(split='test', transform=transform_test)
        elif d == 'voc_val':
            datasets_dict[d] = pascalVOC(split='val', transform=transform_test)
        elif d == 'voc_train':
            if '_' in args.training_set:
                p = float(args.training_set.split('_')[-1])/100
            else:
                p = 1.
            datasets_dict[d] = pascalVOC(split='train', transform=transform_test, p_data=p)
        elif 'nus' in d:
            p, n = utils.nus_decode(args.training_set)
            if n is not None:
                n = None
                p = 0.1
            # print(p,n)
            if d == 'nus_test':
                datasets_dict[d] = nuswide(split='test', img_transform=transform_test, n_data=n, p_data=p)
            elif d == 'nus_val':
                datasets_dict[d] = nuswide(split='val', img_transform=transform_test, n_data=n, p_data=p)
            elif d == 'nus_train':
                datasets_dict[d] = nuswide(split='train', img_transform=transform_test, n_data=n, p_data=p)

    return datasets_dict

def OOD_test(load_path, weight, eval_dataset, inspect_target, model, device, score_fn, args, save_name='OOD_result'):

    split_path = load_path.split('/')
    main_path = '/'.join(split_path[:-1])
    save_dir = f'{main_path}/{save_name}/{split_path[-1]}_{weight.replace(".pt","")}'
    os.makedirs(save_dir, exist_ok=True)
    
    
    if score_fn in IN_SMALLER_LIST:
        pass
    elif score_fn in IN_LARGER_LIST:
       pass
    else:
        return
    
    if '!' in score_fn:
        return

    dataset_color = {
                     'ORIS': 'black',
                     'ORIS_test': 'brown',
                     'coco': 'purple',
                     'coco_val': 'yellow',
                     'coco_test': 'cyan',
                     'DTD':'red',
                     'imagenet':'blue',
                     'voc_test': 'purple',
                     'voc_train': 'orange',
                     'nus_test': 'cyan',
                     'nus': 'cyan',
                     }

    plt.switch_backend('agg')
   
    
    def _score_fn(x):
        
        if score_fn == "Ex":
            with torch.no_grad():
                return model.get_Ex_from_output(x).detach().cpu()
        elif score_fn == "Eybarx_sum":
            with torch.no_grad():
                return  model.get_E_yi_bar_given_x_from_output(x).sum(dim=-1).detach().cpu()
        elif score_fn == "Eyx2_min":
            with torch.no_grad():
                return model.get_Exy2_from_output(x).amin(dim=-1).detach().cpu()
        else:
            print(score_fn)
            exit()
            return None
    
    score_dict = {}
    sqrt = lambda x: int(torch.sqrt(torch.Tensor([x])))
    plot = lambda p, x: tv.utils.save_image(torch.clamp(x, -1, 1), p, normalize=True, nrow=sqrt(x.size(0)))
    model.eval()
    scale_Ex = 1.
    scale_conf = 1.
    dataset_dict = get_dataset(eval_dataset, inspect_target, args)
    
    
    logit_dir = save_dir + '/logit.pt'
    if not os.path.exists(logit_dir):
        logits = {}
        labels = {}
        for ds_ind, dataset_name in enumerate(eval_dataset):
            this_scores = []
            this_logits = []
            dataset = dataset_dict[dataset_name]
            print(dataset_name)
            print(len(dataset))
            dataloader = DataLoader(dataset, batch_size=10, shuffle=False, num_workers=8, drop_last=False)
            if dataset_name == inspect_target:
                gt_his = []
                pre_his = []
            for x, gt in tqdm(dataloader, disable=False, bar_format="{l_bar}{bar:30}{r_bar}"):

                x = x.to(device)
                log = model(x)
                # print(log.amin())
                if dataset_name == inspect_target:
                    gt_his.append(gt)
                    pre_his.append(model.get_prob_from_output(log).detach().cpu())

                if 'BENN' == args.cla_module:  
                    pass
                else:
                    if isinstance(log, dict):
                        this_logits.append({k:v.detach().cpu() for k, v in log.items()})
                    else:
                        this_logits.append(log.detach().cpu())
                scores = _score_fn(log)
                this_scores.extend(scores.numpy())
            if 'BENN' == args.cla_module:  
                pass
            else:
                if isinstance(this_logits[0], dict):
                    keys = this_logits[0].keys()
                    logits[dataset_name] = {k:torch.cat([v[k] for v in this_logits], dim=0) for k in keys}
                else:
                    logits[dataset_name] = torch.cat(this_logits, dim=0)
                
            score_dict[dataset_name] = this_scores
            # print(min(this_scores))
            if dataset_name == inspect_target:
                gt_his = torch.cat(gt_his, dim=0)
                pre_his = torch.cat(pre_his, dim=0)
                _mAP, acc, tpr = utils.get_metrics_eval(pre_his, gt_his)
                result = {
                    'mAP': f'{_mAP:<.5f}',
                    'TPR': f'{tpr:<.5f}',
                    'Acc': f'{acc:<.5f}',
                    
                }
                    
                del gt_his, pre_his
        
        if 'BENN' == args.cla_module:  
            pass
        else:
            pass
            static_dict = {
                'logits': logits,
                'labels': labels,
            }
            torch.save(static_dict, logit_dir)
    else:

        static_dict = torch.load(logit_dir)
        logits = static_dict['logits']
        labels = static_dict['labels']
        for ds_ind, dataset_name in enumerate(eval_dataset):

            log = logits[dataset_name]
            scores = _score_fn(log)
            this_scores = scores.numpy()
            score_dict[dataset_name] = this_scores  
           
           
    # exit() 
    target_ds = inspect_target
    ID_score_list = score_dict[target_ds].copy()

    ID_score_list = torch.tensor(ID_score_list)
    metrics_dict = {}


    for dataset_name, item in score_dict.items():
        print_list = []
        temp_metric = {}
        scores = torch.tensor(item)
        id_score = ID_score_list.clone()

        temp_score = torch.cat([id_score, scores]).numpy()

        temp_true = np.zeros_like(temp_score)

        temp_true[:len(id_score)] += 1
        if score_fn in IN_SMALLER_LIST:
            temp_score = - temp_score
        
        
        FPR95, thr = fpr_and_fdr_at_recall(temp_true, temp_score, pos_label=None)
        # print(FPR95)
        temp_metric['FPR95'] = FPR95
        temp_metric['thershole'] = -thr if score_fn in IN_SMALLER_LIST else thr
       
        AUROC = auroc(torch.from_numpy(temp_score), torch.from_numpy(temp_true))
        temp_metric['AUROC'] = AUROC.item()
        
        AUPRC = auprc(torch.from_numpy(temp_score), torch.from_numpy(temp_true))
        temp_metric['AUPRC'] = AUPRC.item()

        metrics_dict[dataset_name] = temp_metric

    
    ######## Ploting ########
    
    lines = {}
    bin_min = 100000
    bin_max = -100000
    
    fig = plt.figure()
    
    print(score_fn)
    for name, scores in score_dict.items():
        # if "coco" not in name:
            print(f'Name: {name:>10}| FPR95: {metrics_dict[name]["FPR95"]:>10.5f}, AUROC: {metrics_dict[name]["AUROC"]:>10.5f}, AUPRC: {metrics_dict[name]["AUPRC"]:>10.5f}')
            n, bins, _  = plt.hist(scores, bins=100, density=True, alpha=.5, color=dataset_color[name])
            bin_min = min(bins[0], bin_min)
            bin_max = max(bins[-1], bin_max)
            density = ss.gaussian_kde(scores)
            lines[name] = density
            
            plt.axvline(x=metrics_dict[name]["thershole"], color='black')
            

    bins = np.linspace(bin_min, bin_max, 200)
    y_max = 0
    for name, density in lines.items():
        # if "coco" not in name:
        d = density(bins)
        if "voc" in name or "coco" in name or "nus" in name:
            y_max = max(d.max(), y_max)
        plt.plot(bins, d,label=name, color=dataset_color[name])

    save_results_list = ''
    save_results_list_float = ''
    save_results_list_precentage = ''
    save_results_list_tex = ''
    for name in score_dict.keys():
        fpr95_ = metrics_dict[name]["FPR95"]
        auroc_ = metrics_dict[name]["AUROC"]
        auprc_ = metrics_dict[name]["AUPRC"]
        save_results_list_float += f'{name:<10}: [{fpr95_:.5f}, {auroc_:.5f}, {auprc_:.5f}]\n'
        save_results_list_precentage += f'{name:<10}: [{fpr95_*100:.2f}, {auroc_*100:.2f}, {auprc_*100:.2f}]\n'
        save_results_list_tex += f'{name:<10}: ${fpr95_*100:>5.2f} / {auroc_*100:>5.2f} / {auprc_*100:>5.2f}$\n'
    
    save_results_list = 'Float:\n'
    save_results_list += save_results_list_float
    save_results_list += '\n'
    save_results_list += 'Precentage:\n'
    save_results_list += save_results_list_precentage
    save_results_list += '\n'
    save_results_list += 'Tex:\n'
    save_results_list += save_results_list_tex
    
    if 'at' in args.cla_module:
        name_map = {
            'Ex': 'Ex',
            'Eyx2_min': 'min_LWPE'
        }[score_fn]
    elif args.cla_module in ['JE', 'SNoJoE']:
        name_map = {
            'Eybarx_sum': 'joint_energy',
            'Eyx2_min': 'max_logit'
        }[score_fn]
    elif args.cla_module == 'BENN':
        name_map = {
            'Ex': 'EDL',
        }[score_fn]
    results_text_path = os.path.join(save_dir, f'result_{name_map}.txt')

    with open(results_text_path, 'w') as f:
        f.write(save_results_list)
    cell_text = [
        [
            f'{metrics_dict[name]["FPR95"]:>.5f}',
            f'{metrics_dict[name]["AUROC"]:>.5f}', 
            f'{metrics_dict[name]["AUPRC"]:>.5f}',
        ] for name in score_dict.keys()
        ]
    rowLabels = list(score_dict.keys())
    colLabels = [ "FPR95↓","AUROC↑","AUPRC↑"]
    # print(y_max)
    # exit()
    plt.ylim(top=y_max*1.5)
    plt.legend()
    
    # ax2 = fig.add_subplot(2, 1, 2)
    plt.table(cellText=cell_text,
              cellLoc='center',
            rowLabels=rowLabels,
            colLabels=colLabels,
            # loc='bottom',
            bbox=[0., -0.55, 1., 0.4, ],
            )
    # plt.axis('off')
    
    plt.subplots_adjust(left=0.2, bottom=0.35)
    plt.savefig(save_dir + f"/fig_{name_map}_ALL.png")
    
    plt.clf()
    plt.close()
    # pprint(metrics_dict)
        
    
    # exit()
    
    
        