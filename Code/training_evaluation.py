# Author: Fei Xia
# Create Date: July-20-2022
# Last Update: August-10-2023
# If you use any of the datasets or code in this repository for your research, please consider citing our work:
# Xia, F., Kim, K., Eliezer, Y., Shaughnessy, L., Gigan, S., & Cao, H. (2023). Deep Learning with Passive Optical Nonlinear Mapping. arXiv preprint arXiv:2307.08558.


import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from utility import compute_loss

def train_reg(model, device, train_loader, optimizer, epoch):
    '''
    A single training epoch for a regression model.

    Parameters:
    model : nn.Module
        Regression model.
    device : string
        Determines to which device data is moved. Choices: "cuda" or "cpu".
    train_loader : torch.utils.data.DataLoader
        Dataloader containg the training data.
    optimizer : torch.optim.Optimizer
        Optimizer that determines weight update step.
    epoch : int
        Current epoch.

    Returns:
    epoch : int
        Current epoch.
    batch_idx : int
        Number of batches that were processed.
    loss : float
        MSE loss on the last processed training batch (before last update step).
    '''
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = compute_loss(output, target)
        loss.backward()
        optimizer.step()
    return epoch, batch_idx, loss.item()

def test_reg(model, device, test_loader):
    '''
    A single testing epoch for a regression model.

    Parameters:
    model : nn.Module
        Regression model.
    device : string
        Determines to which device data is moved. Choices: "cuda" or "cpu".
    test_loader : torch.utils.data.DataLoader
        Dataloader containg the training data.

    Returns:
    test_loss : float
        Current MSE loss on the testing dataset.
    '''
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = compute_loss(output, target)
            test_loss += loss.item()  # sum up batch loss
    return test_loss
