from torchvision.transforms import v2
from .dataset_helper import *
from .coco_dataset2 import coco as coco2
from torch.utils.data import DataLoader
from .pascalVOC_dataset2 import pascalVOC
from .nus_wide_dataset import nuswide
import torch



AUG_RATIO = 2.0
def get_coco2_dataLoaders(args):

    normalize = v2.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    train_transform = v2.Compose([
        v2.RandomResizedCrop((args.im_size, args.im_size), scale=(1./AUG_RATIO, AUG_RATIO)),
        v2.ClampBoundingBoxes(),
        v2.SanitizeBoundingBoxes(min_size=2, min_area=4),
        v2.RandomHorizontalFlip(),
        # RandAugment(),
        normalize,
    ])
    test_transform = v2.Compose([
        
        v2.Resize((args.im_size, args.im_size)),
        v2.ClampBoundingBoxes(),
        v2.SanitizeBoundingBoxes(min_size=2, min_area=4),
        normalize,
    ])
    
    if '_' in args.training_set:
        p = float(args.training_set.split('_')[-1])/100
    else:
        p = 1.
    
    train_data_ind_path = getattr(args, 'train_data_ind_path', None)
    
    dset_train = coco2(split='train', img_transform=train_transform, included_classes=None, p_data=p)
    dset_val = coco2(split='val', img_transform=test_transform, included_classes=None)
    
    dload_train = DataLoader(dset_train, batch_size=args.batch_size, shuffle=True, num_workers=10, drop_last=True)
    dload_valid = DataLoader(dset_val, batch_size=args.batch_size, shuffle=False, num_workers=4, drop_last=True)
    
    return dload_train, dload_valid

def get_coco2_test_dataLoaders(args):
    normalize = v2.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    test_transform = v2.Compose([
        v2.Resize((args.im_size, args.im_size)),
        normalize,
    ])
    dset_test = coco2(split='test', img_transform=test_transform, included_classes=None)
    dload_test = DataLoader(dset_test, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=False)
    return dload_test

    

def get_pascalVOC_test(args):
    normalize = v2.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    test_transform = v2.Compose([
        v2.Resize((args.im_size, args.im_size)),
        v2.ToImage(), 
        v2.ToDtype(torch.float32, scale=True),
        normalize,
    ])
    if '_' in args.training_set:
        p = float(args.training_set.split('_')[-1])/100
    else:
        p = 1.
    dset_test = pascalVOC(split='test', transform=test_transform, p_data=p)
    dload_test = DataLoader(dset_test, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=False)
    return dload_test 

def get_pascalVOC(args):
    normalize = v2.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    train_transform = v2.Compose([
        v2.RandomResizedCrop((args.im_size, args.im_size), scale=(1./AUG_RATIO, AUG_RATIO)),
        v2.RandomHorizontalFlip(),
        v2.ToImage(), 
        v2.ToDtype(torch.float32, scale=True),
        normalize,
    ])
    test_transform = v2.Compose([
        v2.Resize((args.im_size, args.im_size)),
        v2.ToImage(), 
        v2.ToDtype(torch.float32, scale=True),
        normalize,
    ])
    if '_' in args.training_set:
        p = float(args.training_set.split('_')[-1])/100
    else:
        p = 1.
    
    train_data_ind_path = getattr(args, 'train_data_ind_path', None)
    dset_train = pascalVOC(split='train', transform=train_transform, p_data=p, train_ind_list_path=train_data_ind_path)
    dset_val = pascalVOC(split='val', transform=test_transform, p_data=p)
    dload_train = DataLoader(dset_train, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
    dload_valid = DataLoader(dset_val, batch_size=args.batch_size, shuffle=False, num_workers=4, drop_last=True)

    return dload_train, dload_valid

def get_nus_test(args):
    normalize = v2.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    test_transform = v2.Compose([
        v2.Resize((args.im_size, args.im_size)),
        v2.ToImage(), 
        v2.ToDtype(torch.float32, scale=True),
        normalize,
    ])
    p = None
    n = None
    if '_' in args.training_set:
        ds_args = args.training_set.split('_')[1:]
        for ds_arg in ds_args:
            if ds_arg[-1] == 'p':
                p = float(ds_arg[:-1])/100
            elif ds_arg[-1] == 'n':
                n = int(ds_arg[:-1])
                
    dset_test = nuswide(split='test', img_transform=test_transform, n_data=n, p_data=p)
    dload_test = DataLoader(dset_test, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=False)
    return dload_test 

def get_nus(args):
    normalize = v2.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    train_transform = v2.Compose([
        v2.RandomResizedCrop((args.im_size, args.im_size), scale=(1./AUG_RATIO, AUG_RATIO)),
        v2.RandomHorizontalFlip(),
        v2.ToImage(), 
        v2.ToDtype(torch.float32, scale=True),
        normalize,
    ])
    test_transform = v2.Compose([
        v2.Resize((args.im_size, args.im_size)),
        v2.ToImage(), 
        v2.ToDtype(torch.float32, scale=True),
        normalize,
    ])
    p = None
    n = None
    if '_' in args.training_set:
        ds_args = args.training_set.split('_')[1:]
        for ds_arg in ds_args:
            if ds_arg[-1] == 'p':
                p = float(ds_arg[:-1])/100
            elif ds_arg[-1] == 'n':
                n = int(ds_arg[:-1])

    train_data_ind_path = getattr(args, 'train_data_ind_path', None)
    
    dset_train = nuswide(split='train', img_transform=train_transform, n_data=n, p_data=p, train_ind_list_path=train_data_ind_path)
    dset_val = nuswide(split='val', img_transform=test_transform,  n_data=n, p_data=p)
    dload_train = DataLoader(dset_train, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
    dload_valid = DataLoader(dset_val, batch_size=args.batch_size, shuffle=False, num_workers=4, drop_last=True)

    return dload_train, dload_valid

def get_dataLoaders(args):
    if 'coco' in args.training_set:
        return get_coco2_dataLoaders(args)
    elif 'voc' in args.training_set:
        return get_pascalVOC(args)
    elif 'nus' in args.training_set:
        return get_nus(args)
    

def get_test_dataLoaders(args):
    if 'coco' in args.training_set:
        return get_coco2_test_dataLoaders(args)
    elif 'nus' in args.training_set:
        return get_nus_test(args)
    elif 'voc' in args.training_set:
        return get_pascalVOC_test(args)
    
def get_standard_normalize():
    normalize = v2.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    return normalize

def get_standard_test_transform(args):
    normalize = get_standard_normalize()
    test_transform = v2.Compose([
        v2.Resize((args.im_size, args.im_size)),
        v2.ToImage(), 
        v2.ToDtype(torch.float32, scale=True),
        normalize,
    ])
    return test_transform