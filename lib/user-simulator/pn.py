import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import pickle
import numpy as np
from torch.nn import functional as F
import time
from torch.autograd import gradcheck

def cuda_(var):
    return var.cuda() if torch.cuda.is_available() else var


class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, dim1, output_dim):
        super(PolicyNetwork, self).__init__()
        self.hidden_size = dim1
        self.fc0 = nn.Linear(input_dim, output_dim)
        self.fc1 = nn.Linear(input_dim, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, output_dim)
        self.dropout = nn.Dropout(p=0.2)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.LogSigmoid()
        self.softmax = nn.Softmax(dim=-1)

    def init_weight(self):
        pass

    def forward(self, input_tensor):
        x = self.fc1(input_tensor)
        x = self.dropout(x)
        x = F.relu(x)

        x = self.fc2(x)
        x = self.dropout(x)
        # x = F.relu(x)

        return x