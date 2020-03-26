import torch.nn as nn
import torch.nn.functional as F


import torch.nn as nn


class CNN_3D(nn.Module):
    def __init__(self):
        super(CNN_3D, self).__init__()
        self.group1 = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=(3, 3, 3), padding=(1, 1, 1), stride=(1, 1, 1)),
            nn.BatchNorm3d(32),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 0, 0)))

        self.group2 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1), stride=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 0, 0)),
            )

        self.drop_out = nn.Dropout(0.5)

        self.fc1 = nn.Linear(64 * 2 * 2 * 2, 256)

        self.fc2 = nn.Sequential(
            nn.Linear(256, 2))

        self.softmax = nn.Softmax(dim=0)


    def forward(self, x):
        out = self.group1(x)
        out = self.group2(out)

        out = out.reshape(out.size(0), -1)
        out = self.drop_out(out)
        out = self.fc1(out)
        out = self.fc2(out)

        out = self.softmax(out)
        return out

# class CNN(nn.Module):
#
#     def __init__(self):
#         super(CNN, self).__init__()
#         self.conv1 = nn.Conv3d(in_channels=1, out_channels=32, kernel_size=(3, 3, 3), padding=(1, 1, 1,),
#                                stride=(1, 1, 1), bias=False)
#
#         self.conv2 = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=(3, 3, 3), padding=(1, 1, 1),
#                                stride=(1, 1, 1), bias=False)
#         self.fc1 = nn.Linear(64 * 3 * 3, 256)
#         self.fc2 = nn.Linear(256, 2)
#
#     def forward(self, x):
#         print('Input: ', x.shape)
#         x = self.conv1(x)
#         x = F.relu(x)
#         # print('After conv 1: ',x.shape)
#         x = nn.BatchNorm3d(32)
#         x = F.max_pool3d(x, kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 0, 0))
#
#         x = self.conv2(x)
#         x = F.relu(x)
#         x = nn.BatchNorm3d(64)
#         x = F.max_pool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 0, 0))
#         x = F.dropout3d(0.5)
#         # print('After conv 2: ',x.shape)
#         x = x.reshape(x.size(0), -1)
#         # print('After conv 3: ',x.shape)
#
#         x = F.relu(self.fc1(x))
#         print('After full conv 1: ', x.shape)
#         x = F.softmax(self.fc2(x))
#         print('After full conv 2: ', x.shape)
#         return x

# import torch.nn as nn
# import torch.nn.functional as F
#
#
# class CNN(nn.Module):
#     def _init_(self):
#         super(CNN, self)._init_()
#         self.conv1 = nn.Conv3d(in_channels=1, out_channels=32, kernel_size=(3, 3, 3), padding=(1, 1, 1,),
#                                stride=(1, 1, 1),
#                                bias=False)
#         self.conv2 = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=(3, 3, 3), padding=(1, 1, 1),
#                                stride=(1, 1, 1),
#                                bias=False)
#
#         self.fc1 = nn.Linear(64 * 3 * 3, 256)
#         self.fc2 = nn.Linear(256, 2)
#
#     def forward(self, x):
#         print('Input: ', x.shape)
#         x = self.conv1(x)
#         x = F.relu(x)
#         # print('After conv 1: ',x.shape)
#         x = F.batch_norm(32)
#         x = F.max_pool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 0, 0))
#
#         x = self.conv2(x)
#         x = F.relu(x)
#         x = F.batch_norm(64)
#         x = F.max_pool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 0, 0))
#         x = F.dropout3d(0.5)
#         # print('After conv 2: ',x.shape)
#         x = x.reshape(x.size(0), -1)
#         # print('After conv 3: ',x.shape)
#
#         x = F.relu(self.fc1(x))
#         print('After full conv 1: ', x.shape)
#         x = F.softmax(self.fc2(x))
#         print('After full conv 2: ', x.shape)
#         return x
# import torch.nn as nn
#
#
# class CNN_3D(nn.Module):
#     def __init__(self):
#         super(CNN_3D, self).__init__()
#         self.group1 = nn.Sequential(
#             nn.Conv3d(1, 32, kernel_size=(3, 3, 3), padding=(1, 1, 1), stride=(1, 1, 1)),
#             nn.BatchNorm3d(32),
#             nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 0, 0)),
#             nn.ReLU(inplace=True))
#
#         self.group2 = nn.Sequential(
#             nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1), stride=(1, 1, 1)),
#             nn.BatchNorm3d(64),
#             nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 0, 0)),
#             nn.Dropout3d(p=0.5),
#             nn.ReLU(inplace=True))
#
#
#         self.fc1 = nn.Sequential(
#             nn.Linear(64 * 2 * 2 * 2, 256))
#
#         self.fc2 = nn.Sequential(
#             nn.Linear(256, 2),
#             nn.Softmax())
#
#         self._features = nn.Sequential(
#             self.group1,
#             self.group2
#         )
#
#         self._classifier = nn.Sequential(
#             self.fc1,
#             self.fc2
#         )
#
#     def forward(self, x):
#         out = self._features(x)
#         out = out.contiguous().view(out.size(1),-1)
#         # out = out.contiguous().view(out.size(0), -1)
#         out = self._classifier(out)
#         return self.fc2(out)

