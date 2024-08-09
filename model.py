import torch
import torch.nn as nn
from torchinfo import summary

class Darknet19(nn.Module):
    def __init__(self):
        super(Darknet19, self).__init__()

        # Conv1
        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.act1 = nn.LeakyReLU(0.1, inplace=True)

        # Conv2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.act2 = nn.LeakyReLU(0.1, inplace=True)

        # Conv3
        self.conv3_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3_1 = nn.BatchNorm2d(128)
        self.act3_1 = nn.LeakyReLU(0.1, inplace=True)
        self.conv3_2 = nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3_2 = nn.BatchNorm2d(64)
        self.act3_2 = nn.LeakyReLU(0.1, inplace=True)

        # Conv4
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(128)
        self.act4 = nn.LeakyReLU(0.1, inplace=True)

        # Conv5
        self.conv5_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn5_1 = nn.BatchNorm2d(256)
        self.act5_1 = nn.LeakyReLU(0.1, inplace=True)
        self.conv5_2 = nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn5_2 = nn.BatchNorm2d(128)
        self.act5_2 = nn.LeakyReLU(0.1, inplace=True)

        # Conv6
        self.conv6 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn6 = nn.BatchNorm2d(256)
        self.act6 = nn.LeakyReLU(0.1, inplace=True)

        # Conv7
        self.conv7_1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn7_1 = nn.BatchNorm2d(512)
        self.act7_1 = nn.LeakyReLU(0.1, inplace=True)
        self.conv7_2 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn7_2 = nn.BatchNorm2d(256)
        self.act7_2 = nn.LeakyReLU(0.1, inplace=True)
        self.conv7_3 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn7_3 = nn.BatchNorm2d(512)
        self.act7_3 = nn.LeakyReLU(0.1, inplace=True)
        self.conv7_4 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn7_4 = nn.BatchNorm2d(256)
        self.act7_4 = nn.LeakyReLU(0.1, inplace=True)

        # Conv8
        self.conv8 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn8 = nn.BatchNorm2d(512)
        self.act8 = nn.LeakyReLU(0.1, inplace=True)

        # Conv9
        self.conv9_1 = nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn9_1 = nn.BatchNorm2d(1024)
        self.act9_1 = nn.LeakyReLU(0.1, inplace=True)
        self.conv9_2 = nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn9_2 = nn.BatchNorm2d(512)
        self.act9_2 = nn.LeakyReLU(0.1, inplace=True)
        self.conv9_3 = nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn9_3 = nn.BatchNorm2d(1024)
        self.act9_3 = nn.LeakyReLU(0.1, inplace=True)
        self.conv9_4 = nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn9_4 = nn.BatchNorm2d(512)
        self.act9_4 = nn.LeakyReLU(0.1, inplace=True)

        # Conv10
        self.conv10 = nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn10 = nn.BatchNorm2d(1024)
        self.act10 = nn.LeakyReLU(0.1, inplace=True)

        # Conv11
        self.conv11 = nn.Conv2d(1024, 1000, kernel_size=1, stride=1, padding=0, bias=False)
        self.globalAvgPool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(1000, 3)
        self.fc2 = nn.Linear(1000, 6)

    def forward(self, x):
        # Conv1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)

        # Conv2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)

        # Conv3
        residual = x
        x = self.conv3_1(x)
        x = self.bn3_1(x)
        x = self.act3_1(x)
        x = self.conv3_2(x)
        x = self.bn3_2(x)
        x = self.act3_2(x)
        x += residual
        x = self.act3(x)

        # Conv4
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.act4(x)

        # Conv5
        residual = x
        x = self.conv5_1(x)
        x = self.bn5_1(x)
        x = self.act5_1(x)
        x = self.conv5_2(x)
        x = self.bn5_2(x)
        x = self.act5_2(x)
        x += residual
        x = self.act5(x)

        # Conv6
        x = self.conv6(x)
        x = self.bn6(x)
        x = self.act6(x)

        # Conv7
        residual = x
        x = self.conv7_1(x)
        x = self.bn7_1(x)
        x = self.act7_1(x)
        x = self.conv7_2(x)
        x = self.bn7_2(x)
        x = self.act7_2(x)
        residual = x
        x = self.conv7_3(x)
        x = self.bn7_3(x)
        x = self.act7_3(x)
        x = self.conv7_4(x)
        x = self.bn7_4(x)
        x = self.act7_4(x)
        x += residual
        x = self.act7(x)

        # Conv8
        x = self.conv8(x)
        x = self.bn8(x)
        x = self.act8(x)

        # Conv9
        residual = x
        x = self.conv9_1(x)
        x = self.bn9_1(x)
        x = self.act9_1(x)
        x = self.conv9_2(x)
        x = self.bn9_2(x)
        x = self.act9_2(x)
        residual = x
        x = self.conv9_3(x)
        x = self.bn9_3(x)
        x = self.act9_3(x)
        x = self.conv9_4(x)
        x = self.bn9_4(x)
        x = self.act9_4(x)
        x += residual
        x = self.act9(x)

        # Conv10
        x = self.conv10(x)
        x = self.bn10(x)
        x = self.act10(x)

        # Conv11
        x = self.conv11(x)

        # Global Average Pooling
        x = self


# # Instantiate the model
# model = Darknet19()
# summary(model)