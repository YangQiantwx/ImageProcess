import torch
import torch.nn as nn

class VGG (nn.Module):
    def __init__(self,input_ch = 3,num_classes = 1000):
        super(VGG,self).__init__()
        

        self.features = nn.Sequential (
            nn.Conv2d(input_ch, 64, kernel_size= 3, padding= 1),
            nn.Conv2d(64, 64, kernel_size= 3, padding= 1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size= 3, padding= 1),
            nn.Conv2d(128, 128, kernel_size= 3, padding= 1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size= 3, padding= 1),
            nn.Conv2d(256, 256, kernel_size= 3, padding= 1),
            nn.Conv2d(256, 256, kernel_size= 1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size= 3, padding= 1),
            nn.Conv2d(512, 512, kernel_size= 3, padding= 1),
            nn.Conv2d(512, 512, kernel_size= 1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512, 512, kernel_size= 3, padding= 1),
            nn.Conv2d(512, 512, kernel_size= 3, padding= 1),
            nn.Conv2d(512, 512, kernel_size= 1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.FNN = nn.Sequential (
            nn.Linear(512*7*7,4096),
            nn.ReLU(True),
            nn.Linear(4096,4096),
            nn.ReLU(True),
            nn.Linear(4096, num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self,x):
        x = self.features(x)
        x = x.view(x.size(0),-1)
        x = self.FNN(x)
        return x

model = VGG()
x = torch.rand(1,3,224,224)
print(model.forward(x).shape)



        

