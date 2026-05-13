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
        
        if isinstance(outputs, tuple):
            hidden_states = outputs[0]  #* (batch_size, seq_len, hid_dim)
        else:
            hidden_states = outputs
            
        h_c = self.vec.to(torch.float16)
        
        # proj
        hc_proj = proj_svd(self.v, h_c.squeeze(), self.k) 
        
        # Apply projection to last token's hidden state
        if hidden_states.dim() == 3:
            # For sequence output, modify last token
            modified_hidden = hidden_states.clone()
            modified_hidden[:, -1, :] = self.alpha * hc_proj + hidden_states[:, -1, :]
        else:
            # For pooled output
            modified_hidden = self.alpha * hc_proj + hidden_states
            
        if isinstance(outputs, tuple):
            return (modified_hidden,) + outputs[1:]
        else:
            return modified_hidden