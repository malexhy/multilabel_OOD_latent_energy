import torch
import torch.nn as nn
import numpy as np

    
class resnet_FE(nn.Module):
    def __init__(self, depth=34, in_channel=1, pre_train=True):
        super().__init__()
        if depth not in [18, 34, 50, 101, 152]:
            raise ValueError(f'Expect "deep" be one of [18, 34, 50, 101, 152], got {depth}')
        exec(f'from torchvision.models import resnet{depth} as resnet', globals())
        exec(f'from torchvision.models import ResNet{depth}_Weights as resnet_weights', globals())
        self.in_channel = in_channel

        ### Create feature extructer ###
        if pre_train:
            rm = resnet(weights=resnet_weights.DEFAULT) # type: ignore
        else:
            rm = resnet(weights=None) # type: ignore
                
        feature_extraction = [m for m in list(rm.children())[:-2]]
        out_channel = feature_extraction[-1][-1].bn3.num_features if depth>34 else feature_extraction[-1][-1].bn2.num_features
        self.out_channel = out_channel

        feature_extraction.append(nn.AdaptiveAvgPool2d((1,1)))
        feature_extraction.append(nn.Flatten(start_dim=1))

        if in_channel == 1:
            self.fe = nn.Sequential(
                nn.Conv2d(in_channel,3,1),
                *feature_extraction, 
                )
        elif in_channel == 3:
            self.fe = nn.Sequential(
                *feature_extraction, 
                )
        
    def forward(self, x):
        return self.fe(x)

        
        
        
    
        
        

if __name__ == "__main__":
    from torchinfo import summary
    # torch.manual_seed(1)
    resnet_FE(depth=50)
    
    
    # a = torch.tensor([True], dtype=torch.bool)
    # b = torch.tensor([1+1e-32], dtype=torch.float32)
    