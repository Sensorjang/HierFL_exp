import torch
import torch.nn as nn
import torch.nn.functional as F

class mnist_lenet(nn.Module):

    def __init__(self, input_channels, output_channels):
        super(mnist_lenet, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size= 5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320,50)
        self.fc2 = nn.Linear(50, output_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.conv2_drop(x)
        x = F.max_pool2d(x,2)
        x = F.relu(x)
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return  x

class mnist_cnn(nn.Module):

    # def __init__(self, input_channels, output_channels):
    #     super(mnist_cnn, self).__init__()
    #     self.conv2d_1 = torch.nn.Conv2d(input_channels, 32, kernel_size=3)
    #     self.max_pooling = nn.MaxPool2d(2, stride=2)
    #     self.conv2d_2 = torch.nn.Conv2d(32, 64, kernel_size=3)
    #     self.dropout_1 = nn.Dropout(0.25)
    #     self.flatten = nn.Flatten()
    #     self.linear_1 = nn.Linear(9216, 128)
    #     self.dropout_2 = nn.Dropout(0.5)
    #     self.linear_2 = nn.Linear(128, output_channels)
    #     self.relu = nn.ReLU()
    #     self.softmax = nn.Softmax(dim=1)
    # def forward(self, x):
    #     # x = torch.unsqueeze(x, 1)
    #     x = self.conv2d_1(x)
    #     x = self.relu(x)
    #     x = self.conv2d_2(x)
    #     x = self.relu(x)
    #     x = self.max_pooling(x)
    #     x = self.dropout_1(x)
    #     x = self.flatten(x)
    #     x = self.linear_1(x)
    #     x = self.relu(x)
    #     x = self.dropout_2(x)
    #     x = self.linear_2(x)
    #     # x = self.softmax(self.linear_2(x))
    #     return x
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, 7, padding=3)
        self.act = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.out = nn.Linear(64 * 7 * 7, output_channels)

    def forward(self, x):
        x = x.reshape(-1, 1, 28, 28)
        x = self.pool(self.act(self.conv1(x)))
        x = self.pool(self.act(self.conv2(x)))
        x = x.flatten(1)
        # return self.dense2(self.act(self.dense1(x)))
        return self.out(x)