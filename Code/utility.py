# Author: Fei Xia
# Create Date: July-20-2022
# Last Update: August-10-2023

import math
import numpy as np
import torch

def crazy_reload(package_string):
    exec(rf"import {package_string}")
    exec(rf"reload({package_string})")
    exec(rf"from {package_string} import *", globals())

def compute_loss(output, target, loss_func='mse'):
    if loss_func == 'l1':
        loss = torch.nn.functional.l1_loss(output, target)
    if loss_func == 'mse':
        loss = torch.nn.functional.mse_loss(output, target)
    return loss
