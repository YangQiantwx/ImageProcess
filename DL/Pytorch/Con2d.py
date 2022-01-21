import numpy as np
import torch
import torch.nn as nn

#input size 
input_height = 5
input_width = 5
 
#kernel size
kernel_height = 3
kernel_width = 3
#output size 
output_height =input_height - kernel_height + 1
output_width = input_width - kernel_width + 1
 
input_channel = 3

output_channel = 5
 
input_size = torch.rand([input_channel, input_height, input_width])
kernel_size = torch.rand([output_channel,input_channel, kernel_height, kernel_width])
output_size = torch.rand([output_channel, output_height, output_width])
 
# manual convolution

for output in range(output_channel):
    for input in range(input_channel):
        for i in range(output_height):
            for j in range(output_width):
                kernel_sum = 0
                for m in range(kernel_height):
                    for n in range(kernel_width):
                        output_size[output][i][j] += input_size[input][i + m][j + n] * kernel_size[output][input][m][n]

                        
print('input_size:', input_size.shape)
print('kernel_size:', kernel_size.shape)
print('output_size:', output_size.shape)
