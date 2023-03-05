from turtle import forward
import torch 
import torch.nn as nn 
from networks.blocks import TransformerBlockSingle, ConvBlock 


class TransUnet(nn.Module):
    def __init__(self):
        super().__init__() 
        
        self.enc_conv1 = ConvBlock(2, 16, 3, 2) 
        self.enc_conv2 = ConvBlock(16, 32, 3, 2) 
        self.enc_conv3 = ConvBlock(32, 32, 3, 2) 
        
        self.down = nn.MaxPool3d(3, 2, 1) 
        tf_blks = [] 
        for i in range(4): 
            tf_blks.append(TransformerBlockSingle(32, 2, 2)) 
        self.tfs = nn.Sequential(*tf_blks) 
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        
        self.dec_conv3 = ConvBlock(32 + 32, 32, 3, 1) 
        self.dec_conv2 = ConvBlock(32 + 32, 32, 3, 1) 
        self.dec_conv1 = ConvBlock(32 + 16, 16, 3, 1) 
        self.out_conv = ConvBlock(16, 3, 3, 1) 
        
    def forward(self, x): 
        # encoder 
        e1 = self.enc_conv1(x) 
        e2 = self.enc_conv2(e1) 
        e3 = self.enc_conv3(e2) 
        e4 = self.down(e3) 
        # tf 
        e4 = self.tfs(e4) 
        # decoder 
        d3 = self.dec_conv3(torch.cat([e3, self.up(e4)], dim=1)) 
        d2 = self.dec_conv2(torch.cat([e2, self.up(d3)], dim=1)) 
        d1 = self.dec_conv1(torch.cat([e1, self.up(d2)], dim=1)) 
        # out 
        flow = self.out_conv(self.up(d1)) 
        return flow    