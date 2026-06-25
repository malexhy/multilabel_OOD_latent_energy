from torch.utils import data
from torchvision.transforms import v2
from torchvision import tv_tensors
import torch
from PIL import Image
import json
# import matplotlib.pyplot as plt

SPLIT_DICT={'train':"multi-label-train2014",
            'val':"multi-label-val2014",
            'test:':"multi-label-test2014",}

class coco(data.Dataset):
    def __init__(self,root='./dataset/coco', split="train", img_transform=None, included_classes=None, need_bbox=False, p_data=1.):
        self.root = root
        self.split = split
        self.to_tensor = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
        self.p_data = p_data
        if included_classes is None:
            # included_classes = list(range(80))
            self.n_classes = 80
        else:
            self.n_classes = len(included_classes)
               

        with open(root+'/label.txt', 'r') as f:
            labels = f.readlines()
        
        labels = [l.replace('\n', '') for l in labels]
            
            
        self.img_transform = img_transform
        self.need_bbox = need_bbox

       
        filePath = self.root + f'/{split}2014.json'
        with open(filePath, 'r') as f:
            datafile = json.load(f)
        if (included_classes is None) or (split == 'test'):
            self.GT = list(datafile.values())
            self.Imglist = list(datafile.keys())
            self.label_dict = {i:j for i,j in enumerate(labels)}
        else:
            self.GT = []
            self.Imglist = []
            conver_dict = {j: i for i,j in enumerate(included_classes)}
            self.label_dict = {i:labels[j] for i,j in enumerate(included_classes)}
            for img, gt in datafile.items():
                if len([i for i in gt[0] if i not in included_classes]) or (len(gt[0])==0):
                    continue
                new_label = [conver_dict[i] for i in gt[0]]
                bbox = gt[1]
                gt = [new_label, bbox]
                self.GT.append(gt)
                self.Imglist.append(img)
        
        if p_data < 1.:
            if split == 'train':
                new_n_data = int(len(self.GT) * p_data)
                self.GT = self.GT[:new_n_data]
                self.Imglist = self.Imglist[:new_n_data]
                
                        

    def __len__(self):
        return len(self.Imglist)

    def __getitem__(self, index):
        img = Image.open(self.Imglist[index]).convert("RGB")
        img = self.to_tensor(img)  
        
        if self.split == "test":
            lbl_num = torch.Tensor([-1, -1]).int()
            boxes = torch.Tensor([[0.,0.,0.,0.],[0.,0.,0.,0.]])
        else:
            # print(self.GT[index])
            lbl_num = torch.Tensor(self.GT[index][0]).int()
            boxes = torch.Tensor(self.GT[index][1])
            if len(lbl_num) == 0:
                lbl_num = torch.Tensor([-1, -1]).int()
                boxes = torch.Tensor([[0.,0.,0.,0.],[0.,0.,0.,0.]])
        
        boxes = tv_tensors.BoundingBoxes(boxes,format="XYWH", canvas_size=img.shape[-2:])

        input_dict = {
            'image': img,
            'bbox':boxes,
            "labels":lbl_num,
        }

        if self.img_transform is not None:
            # for _ in range(20):

            output_dict = self.img_transform(input_dict)
            imgs = output_dict['image']
            label = output_dict['labels']
                # if len(label) != 0:
                #     break
        else:
            imgs = img
            label = lbl_num
            output_dict = input_dict
        
            
        bbox = output_dict['bbox']
        # label = target['labels']
        
        lbl = torch.zeros(self.n_classes)
        if len(label) != 0:
            if label[0] != -1:
                lbl[label] = 1
        
        
        lbls = lbl.long()
        
        if self.need_bbox:
            return imgs, lbls, bbox
        else:
            return imgs, lbls
        
        
if __name__ == '__main__':
    pass