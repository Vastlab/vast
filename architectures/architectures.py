import torch.nn as nn
import torchvision.models as models

class LeNet_plus_plus(nn.Module):
    def __init__(self, use_classification_layer=True, use_BG=False, num_classes=10):
        super(LeNet_plus_plus, self).__init__()
        self.conv1_1 = nn.Conv2d(in_channels=1,out_channels=32,kernel_size=(5,5),stride=1,padding=2)
        self.conv1_2 = nn.Conv2d(in_channels=self.conv1_1.out_channels,out_channels=32,kernel_size=(5,5),stride=1,padding=2)
        self.batch_norm1 = nn.BatchNorm2d(self.conv1_2.out_channels)
        self.pool = nn.MaxPool2d(kernel_size=(2, 2),stride=2)
        self.conv2_1 = nn.Conv2d(in_channels=self.conv1_2.out_channels,out_channels=64,kernel_size=(5,5),stride=1,padding=2)
        self.conv2_2 = nn.Conv2d(in_channels=self.conv2_1.out_channels,out_channels=64,kernel_size=(5,5),stride=1,padding=2)
        self.batch_norm2 = nn.BatchNorm2d(self.conv2_2.out_channels)
        self.conv3_1 = nn.Conv2d(in_channels=self.conv2_2.out_channels,out_channels=128,kernel_size=(5,5),stride=1,padding=2)
        self.conv3_2 = nn.Conv2d(in_channels=self.conv3_1.out_channels,out_channels=128,kernel_size=(5,5),stride=1,padding=2)
        self.batch_norm3 = nn.BatchNorm2d(self.conv3_2.out_channels)
        self.fc1 = nn.Linear(in_features=self.conv3_2.out_channels*3*3, out_features=2,bias=True)
        if use_classification_layer:
            if use_BG:
                self.fc2 = nn.Linear(in_features=2, out_features=num_classes+1,bias=True)
            else:
                self.fc2 = nn.Linear(in_features=2, out_features=num_classes, bias=True)
        self.use_classification_layer=use_classification_layer
        self.prelu_act=nn.PReLU()
        
    def forward(self, x):
        x = self.prelu_act(self.pool(self.batch_norm1(self.conv1_2(self.conv1_1(x)))))
        x = self.prelu_act(self.pool(self.batch_norm2(self.conv2_2(self.conv2_1(x)))))
        x = self.prelu_act(self.pool(self.batch_norm3(self.conv3_2(self.conv3_1(x)))))
        x = x.view(-1,self.conv3_2.out_channels*3*3)
        y = self.fc1(x)
        if self.use_classification_layer:
            x = self.fc2(y)
            return x,y
        return y


class LeNet_Square(nn.Module):
    def __init__(self, use_classification_layer=True):
        super(LeNet_Square, self).__init__()
        self.conv1_1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(5, 5), stride=1, padding=2)
        self.conv1_2 = nn.Conv2d(in_channels=self.conv1_1.out_channels, out_channels=32, kernel_size=(5, 5), stride=1,
                                 padding=2)
        self.batch_norm1 = nn.BatchNorm2d(self.conv1_2.out_channels)
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.conv2_1 = nn.Conv2d(in_channels=self.conv1_2.out_channels, out_channels=64, kernel_size=(5, 5), stride=1,
                                 padding=2)
        self.conv2_2 = nn.Conv2d(in_channels=self.conv2_1.out_channels, out_channels=64, kernel_size=(5, 5), stride=1,
                                 padding=2)
        self.batch_norm2 = nn.BatchNorm2d(self.conv2_2.out_channels)
        self.conv3_1 = nn.Conv2d(in_channels=self.conv2_2.out_channels, out_channels=128, kernel_size=(5, 5), stride=1,
                                 padding=2)
        self.conv3_2 = nn.Conv2d(in_channels=self.conv3_1.out_channels, out_channels=128, kernel_size=(5, 5), stride=1,
                                 padding=2)
        self.batch_norm3 = nn.BatchNorm2d(self.conv3_2.out_channels)

        self.conv4_1 = nn.Conv2d(in_channels=self.conv3_2.out_channels, out_channels=256, kernel_size=(5, 5), stride=1,
                                 padding=2)
        self.conv4_2 = nn.Conv2d(in_channels=self.conv4_1.out_channels, out_channels=256, kernel_size=(5, 5), stride=1,
                                 padding=2)
        self.batch_norm4 = nn.BatchNorm2d(self.conv4_2.out_channels)
        self.fc1 = nn.Linear(in_features=self.conv4_2.out_channels * 3 * 3, out_features=2, bias=True)

        if use_classification_layer:
            self.fc2 = nn.Linear(in_features=2, out_features=10, bias=True)
        self.use_classification_layer = use_classification_layer
        self.prelu_act = nn.PReLU()

    def forward(self, x):
        x = self.prelu_act(self.pool(self.batch_norm1(self.conv1_2(self.conv1_1(x)))))
        x = self.prelu_act(self.pool(self.batch_norm2(self.conv2_2(self.conv2_1(x)))))
        x = self.prelu_act(self.pool(self.batch_norm3(self.conv3_2(self.conv3_1(x)))))
        x = self.prelu_act(self.pool(self.batch_norm4(self.conv4_2(self.conv4_1(x)))))
        x = x.view(-1, self.conv4_2.out_channels * 3 * 3)
        y = self.fc1(x)
        if self.use_classification_layer:
            x = self.fc2(y)
            return x, y
        return y


class ResNet(nn.Module):
    def __init__(self,fc_layer_dimension=10,num_of_outputs=10):
        super(ResNet, self).__init__()
        net = models.resnet18(pretrained=False)
        net.fc = nn.Linear(512, fc_layer_dimension)
        self.classifier=nn.Linear(in_features=fc_layer_dimension, out_features=num_of_outputs,bias=False)
#        net.add_module('classifier',nn.Linear(in_features=20, out_features=10,bias=False))
        self.net=net
        
    def forward(self, x):
        fc = self.net(x)
        classifier = self.classifier(fc)
        return classifier,fc
