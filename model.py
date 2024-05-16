# -*- coding: utf-8 -*-


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import BatchNorm2d, Conv1d, Conv2d, ModuleList, Parameter,LayerNorm,InstanceNorm2d

from utils import PHYSICS_LAYER,PHYSICS_DECODER

"""
the parameters:
x-> [batch_num,in_channels,num_nodes,tem_size],
"""

class HSPGCN(nn.Module):
    def __init__(self,c_in,c_out,num_nodes,week,day,recent,K,Kt): 
        super(HSPGCN,self).__init__()
        tem_size=week+day+recent
        self.tem_size = tem_size
        self.week = week
        self.day = day
        self.recent = recent

        self.physics_layer=PHYSICS_LAYER(c_out,c_out,num_nodes,tem_size,K,Kt)
        self.physics_decode=PHYSICS_DECODER(c_out,c_out,num_nodes,tem_size,K,Kt)

        self.bn=BatchNorm2d(c_in,affine=False)

        self.conv1=Conv2d(1,1,kernel_size=(1, 1),padding=(0,0),
                          stride=(1,1), bias=True)
        self.conv2=Conv2d(1,1,kernel_size=(1, 1),padding=(0,0),
                          stride=(1,1), bias=True)
        self.conv3=Conv2d(1,1,kernel_size=(1, 1),padding=(0,0),
                          stride=(1,1), bias=True)
        self.conv4=Conv2d(1,1,kernel_size=(1, 1),padding=(0,0),
                          stride=(1,1), bias=True)
        self.conv5=Conv2d(1,1,kernel_size=(1, 1),padding=(0,0),
                          stride=(1,1), bias=True)
        self.conv6=Conv2d(1,1,kernel_size=(1, 1),padding=(0,0),
                          stride=(1,1), bias=True)
        self.conv7=Conv2d(1,1,kernel_size=(1, 1),padding=(0,0),
                          stride=(1,1), bias=True)
        self.conv8=Conv2d(1,1,kernel_size=(1, 1),padding=(0,0),
                          stride=(1,1), bias=True)
        self.conv9=Conv2d(1,1,kernel_size=(1, 1),padding=(0,0),
                          stride=(1,1), bias=True)
        self.conv10=Conv2d(1,1,kernel_size=(1, 1),padding=(0,0),
                           stride=(1,1), bias=True)

        self.fc1 = torch.nn.Linear(tem_size, 100)
        self.fc2 = torch.nn.Linear(100, 200)
        self.fc3 = torch.nn.Linear(200, 200)
        self.fc4 = torch.nn.Linear(200, tem_size)
        
    def forward(self,x_w,x_w_mask,x_d,x_d_mask,x_r,x_r_mask,train_t_mask,supports):
        x=torch.cat((x_w,x_d,x_r),-1)
        A=supports

        x1 = self.fc1(x)
        x1 = F.relu(x1)
        x1 = self.fc2(x1)
        x1 = F.relu(x1)
        x1 = self.fc3(x1)
        x1 = F.relu(x1)
        x1 = self.fc4(x1)
        aa = F.relu(x1)

        y1,d_adj=self.physics_layer(aa,A,train_t_mask)

        x,_,t_adj,ff =self.physics_decode(y1,A,train_t_mask)


        x1 = x[:,:,:,0:6]
        x2 = x[:, :, :, 6:12]
        x3 = x[:, :, :, 12:18]
        x4 = x[:, :, :, 18:24]
        x5 = x[:, :, :, 24:30]
        x6 = x[:, :, :, 30:36]
        x7 = x[:, :, :, 36:42]
        x8 = x[:, :, :, 42:48]
        x9 = x[:, :, :, 48:54]
        x10 = x[:, :, :, 54:60]

        x1=self.conv1(x1).squeeze(1)
        x2=self.conv2(x2).squeeze(1)
        x3=self.conv3(x3).squeeze(1)
        x4=self.conv1(x4).squeeze(1)#b,n,l
        x5=self.conv2(x5).squeeze(1)
        x6=self.conv3(x6).squeeze(1)
        x7=self.conv1(x7).squeeze(1)
        x8=self.conv2(x8).squeeze(1)#b,n,l
        x9=self.conv3(x9).squeeze(1)
        x10=self.conv4(x10).squeeze(1)#b,n,l

        x=x1+x2+x3+x4+x5+x6+x7+x8+x9+x10

        return x,d_adj,ff,y1


class HSPGCN_L(nn.Module):
    def __init__(self, c_in, c_out, num_nodes, week, day, recent, K, Kt):
        super(HSPGCN_L, self).__init__()
        tem_size = week + day + recent
        self.tem_size = tem_size
        self.week = week
        self.day = day
        self.recent = recent

        self.physics_layer = PHYSICS_DECODER(c_out, c_out, num_nodes, tem_size, K, Kt)
        self.physics_decode = PHYSICS_DECODER(c_out, c_out, num_nodes, tem_size, K, Kt)

        self.bn = BatchNorm2d(c_in, affine=False)

        self.conv1 = Conv2d(1, 1, kernel_size=(1, 1), padding=(0, 0),
                            stride=(1, 1), bias=True)
        self.conv2 = Conv2d(1, 1, kernel_size=(1, 1), padding=(0, 0),
                            stride=(1, 1), bias=True)
        self.conv3 = Conv2d(1, 1, kernel_size=(1, 1), padding=(0, 0),
                            stride=(1, 1), bias=True)
        self.conv4 = Conv2d(1, 1, kernel_size=(1, 1), padding=(0, 0),
                            stride=(1, 1), bias=True)
        self.conv5 = Conv2d(1, 1, kernel_size=(1, 1), padding=(0, 0),
                            stride=(1, 1), bias=True)
        self.conv6 = Conv2d(1, 1, kernel_size=(1, 1), padding=(0, 0),
                            stride=(1, 1), bias=True)
        self.conv7 = Conv2d(1, 1, kernel_size=(1, 1), padding=(0, 0),
                            stride=(1, 1), bias=True)
        self.conv8 = Conv2d(1, 1, kernel_size=(1, 1), padding=(0, 0),
                            stride=(1, 1), bias=True)
        self.conv9 = Conv2d(1, 1, kernel_size=(1, 1), padding=(0, 0),
                            stride=(1, 1), bias=True)
        self.conv10 = Conv2d(1, 1, kernel_size=(1, 1), padding=(0, 0),
                             stride=(1, 1), bias=True)

        self.fc1 = torch.nn.Linear(tem_size, 100)
        self.fc2 = torch.nn.Linear(100, 200)
        self.fc3 = torch.nn.Linear(200, 200)
        self.fc4 = torch.nn.Linear(200, tem_size)

    def forward(self, x_w, x_w_mask, x_d, x_d_mask, x_r, x_r_mask, train_t_mask, supports):
        x_w=self.bn(x_w)
        x_d=self.bn(x_d)
        x_r=self.bn(x_r)
        x = torch.cat((x_w, x_d, x_r), -1)
        A = supports

        x1 = self.fc1(x)
        x1 = F.relu(x1)
        x1 = self.fc2(x1)
        x1 = F.relu(x1)
        x1 = self.fc3(x1)
        x1 = F.relu(x1)
        x1 = self.fc4(x1)
        aa = F.relu(x1)

        y1, d_adj, t_adj, ff = self.physics_layer(aa, A, train_t_mask)
        x, _, t_adj, ff = self.physics_decode(y1, A, train_t_mask)

        x1 = x[:, :, :, 0:6]
        x2 = x[:, :, :, 6:12]
        x3 = x[:, :, :, 12:18]
        x4 = x[:, :, :, 18:24]
        x5 = x[:, :, :, 24:30]
        x6 = x[:, :, :, 30:36]
        x7 = x[:, :, :, 36:42]
        x8 = x[:, :, :, 42:48]
        x9 = x[:, :, :, 48:54]
        x10 = x[:, :, :, 54:60]

        x1 = self.conv1(x1).squeeze(1)
        x2 = self.conv2(x2).squeeze(1)
        x3 = self.conv3(x3).squeeze(1)
        x4 = self.conv1(x4).squeeze(1)  # b,n,l
        x5 = self.conv2(x5).squeeze(1)
        x6 = self.conv3(x6).squeeze(1)
        x7 = self.conv1(x7).squeeze(1)
        x8 = self.conv2(x8).squeeze(1)  # b,n,l
        x9 = self.conv3(x9).squeeze(1)
        x10 = self.conv4(x10).squeeze(1)  # b,n,l

        x = x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8 + x9 + x10

        return x, d_adj, ff, y1

