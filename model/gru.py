''' 
GRU model, fewer number of parameters than LSTM bc fewer number of gates 
'''
import torch 
import torch.nn as nn 
from torch.nn.utils.rnn import pack_padded_sequence