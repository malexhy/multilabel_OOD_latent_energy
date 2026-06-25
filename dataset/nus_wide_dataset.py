import numpy as np
from PIL import Image
import torch
import torchvision
from torch.utils import data
import math
import os
import json
import random


def simple_get_n(data, labels, n):
    return data[:n], labels[:n]

def balance_get_n(data, labels, n, min_n_sample=100):
    n_classes = len(labels[0])
    # min_n_sample = math.ceil(min_ratio*(n/n_classes))
    # print(min_n_sample)
    # n_label_his = np.zeros(n_classes)
    selected_ids = set()
    temp_memory = []
    for c in range(n_classes):
        indices = np.where(labels[:, c] == 1)[0].tolist()
        indices.sort()
        if len(indices) > min_n_sample:
            indices = indices[:min_n_sample]
        temp_memory.append(indices)
        selected_ids.update(indices)
    nm = n-len(selected_ids)
    # print(nm)
    if nm > 0:
        remaining_ids = list(set(range(len(labels))) - selected_ids)
        remaining_ids.sort()
        selected_ids.update(remaining_ids[:nm])
    elif nm < 0:
        while len(selected_ids) > n:
            labels_freq = labels[list(selected_ids)].sum(axis=0).argsort()[::-1]
            for mfl in labels_freq:
                if len(temp_memory[mfl]) > 0:
                    remove_indice = temp_memory[mfl].pop()
                    for i, m in enumerate(temp_memory):
                        if remove_indice in m:
                            temp_memory[i].remove(remove_indice)
                    selected_ids.remove(remove_indice)
                    break
            
    selected_ids = list(selected_ids)
    selected_ids.sort()
    return [data[i] for i in selected_ids], labels[selected_ids]

class nuswide(data.Dataset):
    def __init__(self, root='../data/nus_wide/', split="train",
                 in_dis=True, img_transform=None, label_transform=None, 
                 n_data=None, p_data=None, train_ind_list_path=None):
        self.root = root
        self.split = split
        # self.in_dis = in_dis
        self.n_classes = 81
        self.img_transform = img_transform
        self.label_transform = label_transform
        self.GT = []
        self.Imglist = []
        self.processing(n_data, p_data, train_ind_list_path)

    def processing(self, n_data=None, p_data=None, train_ind_list_path=None):
        if self.split == "train":
            file_img = "nus_wide_train_imglist.txt"
            file_label = "nus_wide_train_label.txt"
        elif self.split == "test":
            file_img = "nus_wide_test_imglist.txt"
            file_label = "nus_wide_test_label.txt"
        elif self.split == "val":
            file_img = "nus_wide_val_imglist.txt"
            file_label = "nus_wide_val_label.txt"
            
        with open(self.root + file_img, 'r') as f:
            img_list = f.readlines()
            
        lbl = np.loadtxt(self.root + file_label, dtype=np.int64)
        select = np.where(np.sum(lbl, axis=1) > 0)[0].tolist()
        select.sort()
        self.GT = lbl[select]
        self.Imglist = [img_list[i].replace('\n', '') for i in select]
        
        if (n_data is not None) and (p_data is not None):
            print(f'Both n_data and p_data provided value. Please use only one.')
            1/0
        
        if self.split == "train":
            if train_ind_list_path is not None:
                target_path = train_ind_list_path + '_nus.txt'
                if os.path.exists(target_path):
                    with open(target_path, 'r') as f:
                        train_ind_list = json.load(f)
                        print('loading')
                else:
                    train_ind_list = list(range(len(self.Imglist)))
                    random.shuffle(train_ind_list)
                    main_path = os.path.dirname(target_path)
                    os.makedirs(main_path, exist_ok=True)
                    print(target_path)
                    print('generation new order')
                    exit()
                    with open(target_path, 'w') as f:
                        json.dump(train_ind_list, f)
                        print('saving')

                self.Imglist = [self.Imglist[ind] for ind in train_ind_list]
                self.GT = self.GT[train_ind_list]
                
        # get_method = simple_get_n
        get_method = balance_get_n
        if n_data is not None:
            self.Imglist, self.GT = get_method(self.Imglist, self.GT, n_data)
            
        if p_data is not None:
            n_data = int(len(self.Imglist) * p_data)
            self.Imglist, self.GT = get_method(self.Imglist, self.GT, n_data)


    def __len__(self):
        return len(self.Imglist)

    def __getitem__(self, index):
        img = Image.open(self.root + self.Imglist[index]).convert('RGB')
        # if self.split == "test":
        #     lbl = -np.ones(self.n_classes)
        # else:
        lbl = self.GT[index]

        if self.img_transform is not None:
            img_o = self.img_transform(img)
            imgs = img_o
        else:
            imgs = img
        if self.label_transform is not None:
            label_o = self.label_transform(lbl)
            lbls = label_o
        else:
            lbls = lbl

        return imgs, lbls
    
    
if __name__ == "__main__":
    from torchvision.transforms import v2
    from dataset_helper import compute_mean_ir
    import glob 
    transfrom = v2.Compose([
        v2.ToImage(), 
        v2.ToDtype(torch.float32, scale=True),
    ])
    for train_data_path in glob.glob(f'./dataset/data_ind/train_ind*nus*'):
        train_data_path = train_data_path.replace('_nus.txt', '')
        # ds = nuswide(split='train', p_data=0.1, train_ind_list_path=train_data_path)
        ds = nuswide(split='train', n_data=500, train_ind_list_path=train_data_path)
        print(ds.GT.sum(axis=0))
        metric = compute_mean_ir(ds.GT)
        print(metric)
        
    # print(len(ds))
    exit()
