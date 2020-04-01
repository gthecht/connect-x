import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

# region Constants
ELITISM = 1
SURVIVAL_THRESHOLD = 0.5
POP_SIZE = 2
MAX_WEIGHT = 30
MIN_WEIGHT = -30
CONNECTION_TYPE = "Convolution"  # could be convolution, pool, FC
RECEPTIVE_FIELD_DEFAULT = 2
# MUTATION_CHANCE = 0.8
# endregion


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 5, 3)  # Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0)
        self.conv2 = nn.Conv2d(5, 10, 3)

        x = torch.rand(16, 16).view(-1, 1, 16, 16)  # reshape so that height and width are (16,16) with changing number of channels
        self.fc_input_shape = None

        self.convs(x)

        self.fc1 = nn.Linear(self.fc_input_shape, 20)
        self.fc2 = nn.Linear(20, 8)

    def convs(self, x):
        # this forwards an input through the convolutional layers
        # And finds fc_input_shape = final convolutional layer output size

        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))

        if self.fc_input_shape is None:
            self.fc_input_shape = x[0].shape[0] * x[0].shape[1] * x[0].shape[2]

        return x

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self.fc_input_shape)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1)


class EvolutionEngine:

    def __init__(self, network_list):
        self.nets = []
        self.next_gen = []
        self.set_nets(network_list)

    def set_nets(self, new_net_list):
        sorted_list = sorted(new_net_list, key=lambda x: x.fitness)
        self.nets = sorted_list
        self.next_gen = []

    def evolve(self):
        self.elitism()

        num_survivors = int(SURVIVAL_THRESHOLD * len(self.nets))
        survivor_list = self.nets[0:num_survivors]

        while len(self.next_gen) < POP_SIZE:
            index = random.randint(num_survivors)
            mutated_offspring = survivor_list[index]
            self.next_gen.append(mutated_offspring)

        return self.next_gen

    def elitism(self):
        for ind in range(ELITISM):
            self.next_gen.append(self.nets[ind])


if __name__ == '__main__':
    net = Net()
    print(net)
    # in_vec = np.random.randint(-1, 1, (4, 4))
