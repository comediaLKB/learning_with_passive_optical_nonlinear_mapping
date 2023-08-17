# Author: Fei Xia
# Create Date: July-20-2022
# Last Update: August-10-2023
# Part of the code was inpsired from repo by Martin Stein
# If you use any of the datasets or code in this repository for your research, please consider citing our work:
# Xia, F., Kim, K., Eliezer, Y., Shaughnessy, L., Gigan, S., & Cao, H. (2023). Deep Learning with Passive Optical Nonlinear Mapping. arXiv preprint arXiv:2307.08558.

import numpy as np
import torch
from data_processing import process_data, create_data_loaders, CavityDataset
from model_definition import MLP
from training_evaluation import train_reg, test_reg
from training_loop import train_and_search

def main(input, output):
    # Process data
    input_processed, output_processed = process_data(input, output)
    train_loader, test_loader = create_data_loaders(input_processed, output_processed)

    # Define model
    input_dim = input_processed.shape[1]
    output_dim = output_processed.shape[1]
    model = MLP(input_dim, output_dim, hidden_layers=[100, 100], use_batchnorm=True, activation_function='relu')

    # Set hyperparameters and optimizer
    lr = 0.001
    epochs = 100
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Train model
    dt_path = "trained_model.pt"
    test_loss, dt_path = train_and_search(
        train_loader,
        test_loader,
        dt_path,
        input_dim,
        output_dim,
        MLP,
        lr,
        device='cuda',
        epochs=epochs
    )

    # Evaluate model
    test_loss = test_reg(model, 'cuda', test_loader)

    # Save model
    torch.save(model, dt_path)

    return test_loss

if __name__ == '__main__':
    # Load input and output data (replace with actual data loading code)
    input = np.random.rand(1000, 28, 28)
    output = np.random.rand(1000, 128, 128, 2)

    # Run main function
    test_loss = main(input, output)
    print(f"Test loss: {test_loss}")
