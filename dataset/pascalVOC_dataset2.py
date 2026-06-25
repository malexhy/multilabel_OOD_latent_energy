# import torchvision
from torch.utils import data
import torch
from scipy.io import loadmat
from PIL import Image
import json
import os
import random
import numpy as np
from dataset_helper import compute_mean_ir

def weighted_shuffle_ind(weights):
    ind = list(range(len(weights)))
    scores = [(random.random() ** (1.0 / w), i) for i, w in zip(ind, weights)]
    scores.sort(reverse=True, key=lambda x: x[0])
    return [item for score, item in scores]

        
class pascalVOC(data.Dataset):
    def __init__(
        self, 
        root='../data', 
        split='train', 
        transform=None, 
        p_data=1., 
        train_ind_list_path=None,
        shuffle_method=0) -> None:
        super().__init__()
        
        data_dict = {0: 'aeroplane', 1: 'bicycle', 2: 'bird', 3: 'boat', 4: 'bottle', 5: 'bus', 6: 'car', 7: 'cat',
                        8: 'chair', 9: 'cow', 10: 'diningtable', 11: 'dog', 12: 'horse', 13: 'motorbike', 14: 'person',
                        15: 'pottedplant', 16: 'sheep', 17: 'sofa', 18: 'train', 19: 'tvmonitor'}
        
        self.data_dict_inver = {b:a for a, b in data_dict.items()}
        split_file = root+f'/VOC_split/voc12-{split}.mat'
        datafile = loadmat(split_file)
        self.n_classes = len(data_dict)
        if split == "test":
            self.GT = None
        else:
            self.GT = datafile['labels']
        self.Imglist = datafile['Imlist']
        if split == "test":
            self.Imglist = [img.replace('./Pascal/VOCdevkit/VOC2012', f'{root}/voc_test_2/VOC2012_test').strip() for img in self.Imglist]
        else:
            self.Imglist = [img.replace('./Pascal', f'{root}').strip() for img in self.Imglist]
        # print(self.Imglist[0])
        # exit()
        self.transfrom = transform
        
        if split == "train":

            if train_ind_list_path is not None:
                target_path = train_ind_list_path + '_voc.txt'
                if os.path.exists(target_path):
                    with open(target_path, 'r') as f:
                        train_ind_list = json.load(f)
                        print(f'loading from {target_path}')
                        # exit()
                else:
                    if shuffle_method == 0:
                        train_ind_list = list(range(len(self.Imglist)))
                        random.shuffle(train_ind_list)
                    elif shuffle_method == 1:
                        offset = 20
                        label_weight = -((np.arange(self.n_classes) + offset+1)/(self.n_classes+offset))**1
                        np.random.shuffle(label_weight)
                        label_weight[14] = label_weight.max()/4
                        # print(label_weight)
                        weight = self.GT @ label_weight
                        train_ind_list = weighted_shuffle_ind(weight)
                    main_path = os.path.dirname(target_path)
                    os.makedirs(main_path, exist_ok=True)
                    print(f'saving to {target_path}')
                    exit()
                    with open(target_path, 'w') as f:
                        json.dump(train_ind_list, f)

                    
                self.Imglist = [self.Imglist[ind] for ind in train_ind_list]
                self.GT = self.GT[train_ind_list]
            else:
                print('Using original order!')
        else:
            pass
        if p_data < 1.:
            n_data = int(len(self.Imglist) * p_data)
            if split == "train":                    
                self.Imglist = self.Imglist[:n_data]
                self.GT = self.GT[:n_data]
        
        if split == 'train':
            print(f"Training set's MeanIR = {compute_mean_ir(self.GT):.2f}")
   
    
        
    def __len__(self):
        return len(self.Imglist)
    
    def __getitem__(self, index):
        img = Image.open(self.Imglist[index].strip()).convert('RGB')
        if self.GT is not None:
            lbl = self.GT[index]
        else:
            lbl = torch.zeros(self.n_classes, dtype=torch.long)
            
        if self.transfrom is not None:
            img_o = self.transfrom(img)
            imgs = img_o
        else:
            imgs = img
            
        lbls = lbl
        # print(type(lbls))
        return imgs, lbls

    
if __name__ == '__main__':

    from pprint import pprint
    import glob
        
    p = 0.1
    voc_ds = pascalVOC(split='train', p_data=p, train_ind_list_path=None)
    mean_ir = compute_mean_ir(voc_ds.GT)
    print(mean_ir)
    seed = None
    np.random.seed(seed)
    random.seed(seed)
    
    for j in range(1):
        # for train_data_path in glob.glob(f'./dataset/data_ind/train_ind_4_*voc*'):
        for train_data_path in sorted(glob.glob(f'./dataset/data_ind/train_ind*voc*')):
            train_data_path = train_data_path.replace('_voc.txt', '')
            voc_ds = pascalVOC(split='train', p_data=p, train_ind_list_path=train_data_path, shuffle_method=1)
            mean_ir = compute_mean_ir(voc_ds.GT)
            print(mean_ir)
