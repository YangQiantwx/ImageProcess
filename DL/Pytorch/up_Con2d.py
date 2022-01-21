import numpy as np
import torch
import torch.nn as nn

def Con2d(tensor,kernel,output_channel):
    #Generate
    image_size = tensor.shape
    kernel_size = kernel.shape
    #input size 
    input_channel = image_size[0]
    input_height = image_size[1]
    input_width =  image_size[2]    
    #kernel size
    kernel_height = kernel_size[2]
    kernel_width = kernel_size[3]

    output_height =input_height - kernel_height + 1
    output_width = input_width - kernel_width + 1

    output_size = torch.empty([output_channel, output_height, output_width])
    
    for output in range(output_channel):
       for input in range(input_channel):
          for i in range(output_height):
              for j in range(output_width):
                 kernel_sum = 0
                 for m in range(kernel_height):
                    for n in range(kernel_width):
                        output_size[output][i][j] += tensor[input][i + m][j + n] * kernel[output][input][m][n]
    return output_size

     
input_channel = 3
output_channel = 10
tensor = torch.rand([input_channel, 5, 5])
kernel = torch.rand([output_channel, input_channel, 2, 2])
print(Con2d(tensor,kernel,output_channel))
print(Con2d(tensor,kernel,output_channel).shape)












