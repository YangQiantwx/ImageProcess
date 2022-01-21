import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo


# __all__ = [
#     'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
#     'vgg19_bn', 'vgg19',
# ]


# model_urls = {
#     'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
#     'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
#     'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
#     'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
#     'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
#     'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
#     'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
#     'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
# }



# VGG:

# 1. convolutional layers
# 2. FCNN
# 3. Forward
# 4. Backward


# class conv2d():
#     def __init__(self, input_channels, output_channels, kernel_size, padding):
#         self.w = (input_channels, output_channels, kernel_size, kernel_size)
#         self.b = (input_channels, output_channels)
#     def forward(self, x):
#         return self.w * x + self.b
#     def backward(self):
#         grad_w = ####
#         grad_b = ####
#         return grad_w, grad_b

class VGG(nn.Module):

    def __init__(self, num_classes=1000, init_weights=True):
        super(VGG, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            )

        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Linear(4096, num_classes),
            nn.Softmax(num_classes)
        )

    def forward(self, x): # 16 3 224 224
        x = self.features(x) # 16 512 7 7 - > 16, 512*7*7
        x = x.view(x.size(0), -1) # x = torch.reshape(x, (16, 512*7*7))
        x = self.classifier(x) # 16 1000
        return x

    # def backward(self, x):
model = VGG()
x = torch.rand(1,3,224,224)
print(model(x))