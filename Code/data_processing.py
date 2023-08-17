# Author: Fei Xia
# Create Date: July-20-2022
# Last Update: August-10-2023
# If you use any of the datasets or code in this repository for your research, please consider citing our work:
# Xia, F., Kim, K., Eliezer, Y., Shaughnessy, L., Gigan, S., & Cao, H. (2023). Deep Learning with Passive Optical Nonlinear Mapping. arXiv preprint arXiv:2307.08558.

import numpy as np
import torch
from sklearn.model_selection import train_test_split

def process_data(input, output, N):
    input = torch.from_numpy(np.reshape(input, (input.shape[0], N*N)).astype('float32')).cuda()
    output = torch.from_numpy(np.reshape(output[:, :, :, 1], (output.shape[0], 128*128)).astype('float32')).cuda()
    return input, output

def create_data_loaders(input, output, batch_size=100, test_batch_size=100):
    data_train, data_test, y_train, y_test = train_test_split(input, output, test_size=0.1)
    train_sampler = torch.utils.data.BatchSampler(
        torch.utils.data.RandomSampler(range(data_train.shape[0])),
        batch_size=batch_size, drop_last=False
    )
    test_sampler = torch.utils.data.BatchSampler(
        torch.utils.data.SequentialSampler(range(data_test.shape[0])),
        batch_size=test_batch_size, drop_last=False
    )
    train_loader = torch.utils.data.DataLoader(
        CavityDataset(data_train/255., y_train/y_train.max()),
        batch_size=batch_size
    )
    test_loader = torch.utils.data.DataLoader(
        CavityDataset(data_test/255., y_test/y_test.max()),
        batch_size=batch_size
    )
    return train_loader, test_loader

class CavityDataset(torch.utils.data.Dataset):
    def __init__(self, data_import, targets):
        self.data = data_import
        self.targets = targets

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        data = self.data[idx]
        return data, self.targets[idx]
