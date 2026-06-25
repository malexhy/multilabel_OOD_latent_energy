from torch import nn
import torch
import math

    
class print_shape(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    def forward(self, x):
        print(x.shape)
        return x
    
class reshape(nn.Module):
    def __init__(self, out_shape):
        super().__init__()
        self.out_shape = out_shape
        
    def forward(self, x):
        return x.view(-1, *self.out_shape)

class permute(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.dims = dims
    def forward(self, inpt:torch.Tensor):
        return inpt.permute(*self.dims)
    
class n_linear_base(nn.Module):
    def __init__(self, in_features, out_features, repeat_n, bias=True, device=None, dtype=None, *args, **kwargs) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.weight = nn.Parameter(torch.empty((repeat_n, in_features, out_features), **factory_kwargs))
        self.bias = nn.Parameter(torch.empty((repeat_n, out_features), **factory_kwargs)) if bias else None
        self.reset_parameters()
        
    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, mode='fan_in')
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
            
class up_scale_n_linear(n_linear_base):
    def __init__(self, in_features, out_features, repeat_n, bias=True, *args, **kwargs) -> None:
        super().__init__(in_features, out_features, repeat_n, bias, *args, **kwargs)
        if self.bias is None:
            self.forward_fn = self.forward_no_bn
        else:
            self.forward_fn = self.forward_bn
            
    def forward_no_bn(self, inpt):
        return torch.einsum('bl,nlk->bnk', inpt, self.weight)
    
    def forward_bn(self, inpt):
        # print(inpt.shape)
        # print(self.weight.shape)
        return self.forward_no_bn(inpt) + self.bias
        
    def forward(self, inpt):
        # inpt (b, in_f)
        return self.forward_fn(inpt)

    
class ind_n_linear(n_linear_base):
    def __init__(self, in_features, out_features, repeat_n, bias=True, *args, **kwargs) -> None:
        super().__init__(in_features, out_features, repeat_n, bias, *args, **kwargs)
        if self.bias is None:
            self.forward_fn = self.forward_no_bn
        else:
            self.forward_fn = self.forward_bn
    
    def forward_no_bn(self, inpt):
        return torch.einsum('bnl,nlk->bnk', inpt, self.weight)
    
    def forward_bn(self, inpt):
        return self.forward_no_bn(inpt) + self.bias
        
    def forward(self, inpt):
        # inpt (b, n, in_f)
        return self.forward_fn(inpt)       

        
        
        