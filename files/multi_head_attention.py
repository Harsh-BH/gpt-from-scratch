import torch
import torch.nn as nn
import torch.nn.functional as F

class Multiheadattention(nn.module){
    def __init__(self,d_model,num_heads):
        super(Multiheadattention,self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % num_heads == 0

        self.W_q = nn.Linear(d_model,d_model)
        self.W_k = nn.Linear(d_model,d_model)
        self.W_v = nn.Linear(d_model,d_model)
        self.fc_out = nn.Linear(d_model,d_model)

    

}