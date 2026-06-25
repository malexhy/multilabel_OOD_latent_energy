import os
import torch
import numpy as np
from torchvision import datasets as datasets
from torcheval.metrics.functional import multilabel_auprc as mAP
from torcheval.metrics.functional import multilabel_accuracy as Acc
import json
from glob import glob
import shutil


recall_level_default = 0.95

def find_best(load_path):
    return find_best_result(load_path, 'mAP')

def get_value(str_log: str, target):
    log_data = str_log.strip().replace(' ', '').split(',')
    for ld in log_data:
        if target in ld:
            value = float(ld.replace(f'{target}:', ''))
            return value
    return 0
        
def find_best_result(load_path, target='AUC'):
    log_path = load_path + '/save_log.json'
    with open(log_path, 'r') as f:
        metric = json.load(f)
        best = max(metric, key=lambda x: get_value(metric.get(x), target))
        # print(best)
    if 'ema' not in best:
        for k in metric.keys():
            if 'ema' in k:
                value = get_value(metric[k], target=target)
                if value == get_value(metric[best], target=target):
                    best = k
    return best

def TPR(inpt: torch.Tensor, target: torch.Tensor, output_fpr=False):
    inpt = (inpt > 0.5).to(torch.float)
    target = target.to(torch.bool)
    tpr = inpt[target].mean()
    if output_fpr:
        fpr = (1-inpt)[torch.logical_not(target)].mean()
        return tpr, fpr
    else:
        return tpr

def get_metrics_eval(prob, target):
    n_classes = prob.shape[-1]
    _mAP = mAP(prob, target, num_labels=n_classes).item()
    acc = Acc(prob, target, criteria="hamming").item()
    tpr = TPR(prob, target).item()
    return _mAP, acc, tpr

def stable_cumsum(arr, rtol=1e-05, atol=1e-08):
    """Use high precision for cumsum and check that final value matches sum
    Parameters
    ----------
    arr : array-like
        To be cumulatively summed as flat
    rtol : float
        Relative tolerance, see ``np.allclose``
    atol : float
        Absolute tolerance, see ``np.allclose``
    """
    out = np.cumsum(arr, dtype=np.float64)
    expected = np.sum(arr, dtype=np.float64)
    if not np.allclose(out[-1], expected, rtol=rtol, atol=atol):
        raise RuntimeError('cumsum was found to be unstable: '
                           'its last element does not correspond to sum')
    return out
 
def fpr_and_fdr_at_recall(y_true, y_score, recall_level=recall_level_default, pos_label=None):
    classes = np.unique(y_true)
    if (pos_label is None and
            not (np.array_equal(classes, [0, 1]) or
                     np.array_equal(classes, [-1, 1]) or
                     np.array_equal(classes, [0]) or
                     np.array_equal(classes, [-1]) or
                     np.array_equal(classes, [1]))):
        raise ValueError("Data is not binary and pos_label is not specified")
    elif pos_label is None:
        pos_label = 1.

    # make y_true a boolean vector
    y_true = (y_true == pos_label)

    # sort scores and corresponding truth values
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]


    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    tps = stable_cumsum(y_true)[threshold_idxs]
    fps = 1 + threshold_idxs - tps      

    thresholds = y_score[threshold_idxs]

    recall = tps / tps[-1]

    last_ind = tps.searchsorted(tps[-1])
    sl = slice(last_ind, None, -1)      
    recall, fps, tps, thresholds = np.r_[recall[sl], 1], np.r_[fps[sl], 0], np.r_[tps[sl], 0], thresholds[sl]

    cutoff = np.argmin(np.abs(recall - recall_level))
    if np.array_equal(classes, [1]):
        return 0.5, thresholds[cutoff]  

    return fps[cutoff] / (np.sum(np.logical_not(y_true))), thresholds[cutoff]

def remove_non_best(path, best_ckpt, confirm=False):

    all_ckpt = glob(path+'/*ckpt*')
    for pt in all_ckpt:
        best_ckpt_path = os.path.join(path, best_ckpt)
        if pt != best_ckpt_path:
            if 'backup' in pt:
                print('Found backup, skipping...')
                continue
            print(f'Removing {pt}')
            print(f'Remove ckpt: {os.path.basename(pt)}')
            if confirm:
                if os.path.isdir(pt):
                    shutil.rmtree(pt)
                elif os.path.isfile(pt):
                    os.remove(pt)
                else:
                    1/0

def nus_decode(ds_name):
    p = None
    n = None
    sd_args = ds_name.split('_')[1:]
    for ds_arg in sd_args:
        if ds_arg[-1] == 'p':
            p = float(ds_arg[:-1])/100
        elif ds_arg[-1] == 'n':
            n = int(ds_arg[:-1])
    return p, n

if __name__ == '__main__':
    pass
    
    