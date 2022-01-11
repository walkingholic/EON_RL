import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.nn.init
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 랜덤 시드 고정
torch.manual_seed(777)

# GPU 사용 가능일 경우 랜덤 시드 고정
if device == 'cuda':
    torch.cuda.manual_seed_all(777)

learning_rate = 0.001
training_epochs = 15
batch_size = 100

class CNN(torch.nn.Module):

    def __init__(self):
        super(CNN, self).__init__()


        # 첫번째층
        # ImgIn shape=(?, 28, 28, 1)        (N-F)/stride+1
        #    Conv     -> (?, 28, 28, 32)
        #    Pool     -> (?, 14, 14, 32)
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))

        # 두번째층
        # ImgIn shape=(?, 14, 14, 32)
        #    Conv      ->(?, 14, 14, 64)
        #    Pool      ->(?, 7, 7, 64)
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer4 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))

        # self.layer5 = torch.nn.Sequential(
        #     torch.nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
        #     torch.nn.ReLU(),
        #     torch.nn.MaxPool2d(kernel_size=2, stride=2))

        self.Covnet = torch.nn.Sequential(
            self.layer1, self.layer2, self.layer3, self.layer4
        )

        # 전결합층 7x7x64 inputs -> 10 outputs
        self.fc = torch.nn.Linear(1152+62, 15, bias=True)

        # 전결합층 한정으로 가중치 초기화
        torch.nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, x):
        (img, info) = x
        # print('img', img.size())
        # print('info', info.size())
        # out = self.layer1(img)
        # print('l1', out.size())
        # out = self.layer2(out)
        # print('l2',out.size())
        # out = self.layer3(out)
        # print('l3', out.size())
        # out = self.layer4(out)
        # print('l4', out.size())

        out = self.Covnet(img)

        out = out.view(out.size(0), -1)   # 전결합층을 위해서 Flatten
        print('view', out.size())
        out = torch.cat([out, info], dim=1)
        print('cat', out.size())
        out = self.fc(out)
        print(out.size())
        return out
#
# x = torch.randn(1,1,100,100, )
# print(x.size())
# model = CNN().to(device)
# model(x)
#

