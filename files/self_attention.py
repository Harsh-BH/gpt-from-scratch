import numpy as np

def softmax(x, axis):
    """
    Compute softmax values for each sets of scores in x.
    """
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / e_x.sum(axis=axis, keepdims=True)

def self_attention(Q, K, V):
   """
   conpute self attention adn return the output
   """

   d_k = Q.shape[-1]
   scores = np.matmul(Q, K.T) / np.sqrt(d_k)
   attention = softmax(scores, axis=-1)
   output = np.matmul(attention, V)
   return output , attention

Q = np.random.rand(3, 4)
K = np.random.rand(3, 4)
V = np.random.rand(3, 4)

oup, att = self_attention(Q, K, V)
print(oup)





