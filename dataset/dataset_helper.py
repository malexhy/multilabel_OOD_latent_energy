from typing import Dict, Type
from torchvision.transforms import v2
import numpy as np

def compute_mean_ir(Y):
    Y = np.asarray(Y, dtype=np.int32)
    N, C = Y.shape
    label_counts = np.sum(Y, axis=0)
    max_count = np.max(label_counts)

    with np.errstate(divide='ignore'):
        ir_lbl = max_count / label_counts
        ir_lbl[label_counts == 0] = np.inf 
        
    finite_ir = ir_lbl[np.isfinite(ir_lbl)]
    mean_ir = np.mean(finite_ir) if len(finite_ir) > 0 else np.inf
        
    return mean_ir


class RandAugment():
    def __init__(self) -> None:
        
        self.augment = v2.RandAugment()
        
    def __call__(self, inpt_dict:Dict):
        if 'image' in inpt_dict:
            img = inpt_dict['image']
            augmented_image = self.augment(img)
            inpt_dict['image'] = augmented_image
        return inpt_dict




        
        