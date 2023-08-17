
# Author: Fei Xia

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from training_evaluation import train_reg, test_reg

def train_and_search(
        train_loader,
        test_loader,
        model_path,
        input_dim,
        output_dim,
        ModelClass,
        lr,
        device,
        scheduler=None,
        step_size=50,
        gamma=0.5,
        epochs=300,
        train_ratio=0.9,
        trial=None,
        **model_args):
    '''
    Training loop for regression model.

    Trains ModelClass for a given number of epochs on training data provided in train_loader with a step learning rate
    scheduler. The trained model is saved at model_path and the final testing loss and the model's location is returned.

    Parameters:
    train_loader : torch.utils.data.DataLoader
        Dataloader containg the training data.
    test_loader : torch.utils.data.DataLoader
        Dataloader containg the testing data.
    model_path : string
        Path to file in which trained model will be saved.
    input_dim : int
        Number of inputs to the experiment per sample.
    output_dim : int
        Number of outputs from the experiment per sample.
    ModelClass : nn.Module
        ModelClass architecture for digital model. All parameters needed to specify ModelClass completely can be passed
        through **kwargs.
    lr : float
        Learning rate.
    device : string
        Determines to which device data is moved. Choices: "cuda" or "cpu".

    Returns:
    test_loss : float
        Final testing loss.
    model_path : string
        Path to the saved trained model.
    '''
    model = ModelClass(input_dim, output_dim, **model_args).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    if not scheduler:
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

    train_ls = []
    test_ls = []

    for epoch in range(1, epochs + 1):
        epoch, batch_idx, loss = train_reg(model, device, train_loader, optimizer, epoch)
        test_loss = test_reg(model, device, test_loader)
        scheduler.step()
        train_ls.append(loss)
        test_ls.append(test_loss)

    torch.save(model, model_path)
    np.savez(model_path,
            train_ls=train_ls,
            test_ls=test_ls,
            model_args=model_args)

    del model
    del loss
    del optimizer
    del scheduler

    return test_loss, model_path
