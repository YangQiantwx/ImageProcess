import torch
import torch.nn as nn

class LeNET (nn.Module):
    def __init__(self,num_classes = 10):
        super(LeNET,self).__init__()
        

        self.features = nn.Sequential (
            nn.Conv2d(1, 6, kernel_size= 5),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size= 5),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.FNN = nn.Sequential (
            nn.Linear(16*5*5,120),
            nn.ReLU(True),
            nn.Linear(120,84),
            nn.ReLU(True),
            nn.Linear(84, num_classes),
           
        )

    def forward(self,x):
        x = self.features(x)
        x = x.view(x.size(0),-1)
        x = self.FNN(x)
        return x

model = LeNET()
x = torch.rand(16,1,32,32)
print(model.forward(x))
print(model.forward(x).shape)



        

