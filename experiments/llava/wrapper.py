import torch
import torch.nn as nn

def proj_svd(v, y, k):
    """
    project y into the subspace spanned by top k singular vectors v[:, :k]
    """
    v = v.to(torch.float16)
    y = y.to(torch.float16)
    dot_products = torch.matmul(v[:, :k].T, y)
    projection = torch.matmul(v[:, :k], dot_products)
    
    return projection

class Wrapper(nn.Module):
    def __init__(
        self,
        block,
        vec,
        v,
        k = 20,
        alpha = 2.0,  #! hyperparameters
    ):
        super(Wrapper, self).__init__()
        self.block = block
        self.vec = vec  #* (hid_dim,)
        self.v = v
        self.k = k
        self.alpha = alpha
        
    def forward(self, *args, **kwargs): 
        outputs = self.block(*args, **kwargs)
        hidden_states = outputs[0]  #* (batch_size, seq_len, hid_dim)
        h_c = self.vec.to(torch.float16)
        
        # proj
        hc_proj = proj_svd(self.v, h_c.squeeze(), self.k) 
       
        return (self.alpha*hc_proj+hidden_states, *outputs[1:])
            

