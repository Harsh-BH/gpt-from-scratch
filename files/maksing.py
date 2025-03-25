import numpy as np
import torch.nn.functional as F
import torch


def padding_mask(seq, pad_token=0):
    return (seq != pad_token).astype(np.float32)

def look_ahead_mask(size):

    mask = np.triu(np.ones((size, size)), k=1)
    mask = mask.astype(np.float32)
    mask[mask == 1] = -np.inf
    
    return mask



def apply_mask(mask, scores):

    masked_scores = scores + mask
    return F.softmax(masked_scores , dim=-1)

scores = torch.rand(3, 3)
mask = torch.tensor(look_ahead_mask(3))

print("Before Masking:\n", scores)
masked_scores = apply_mask(scores, mask)
print("After Masking:\n", masked_scores)