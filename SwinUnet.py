import mindspore  
import mindspore.nn as nn 
from .blocks import TransformerBlockSingle, ConvBlock 


class SwinUnet(nn.Cell): 
    def __init__(self):
        super().__init__() 
        
        self.enc_conv1 = ConvBlock(2, 16, 3, 2) 
        self.enc_tf1 = TransformerBlockSingle(16, 4, 4) 
        self.enc_conv2 = ConvBlock(16, 32, 3, 2) 
        self.enc_tf2 = TransformerBlockSingle(32, 2, 2) 
        self.enc_conv3 = ConvBlock(32, 32, 3, 2) 
        self.enc_tf3 = TransformerBlockSingle(32, 1, 1) 
        
        self.mid_conv = ConvBlock(32, 32, 3, 2) 
        # self.mid_tf = TransformerBlockSingle(32, 2, 1) 
        self.up = nn.Upsample(scale_factor=2, mode='nearest') 
        
        self.dec_conv3 = ConvBlock(32 + 32, 32, 3, 1) 
        self.dec_tf3 = TransformerBlockSingle(32, 1, 1) 
        self.dec_conv2 = ConvBlock(32 + 32, 32, 3, 1) 
        self.dec_tf2 = TransformerBlockSingle(32, 2, 2) 
        self.dec_conv1 = ConvBlock(32 + 16, 16, 3, 1) 
        self.dec_tf1 = TransformerBlockSingle(16, 4, 4) 
        self.out_conv = ConvBlock(16, 3, 3, 1) 
        
    def construct(self, x): 
        # encoder 
        e1 = self.enc_conv1(x) 
        e1 = self.enc_tf1(e1) 
        e2 = self.enc_conv2(e1) 
        e2 = self.enc_tf2(e2) 
        e3 = self.enc_conv3(e2) 
        e3 = self.enc_tf3(e3) 
        # mid 
        mid = self.mid_conv(e3) 
        # mid = self.mid_tf(mid) 
        # decoder 
        d3 = self.dec_conv3(mindspore.cat([e3, self.up(mid)], dim=1)) 
        d2 = self.dec_conv2(mindspore.cat([e2, self.up(d3)], dim=1)) 
        d1 = self.dec_conv1(mindspore.cat([e1, self.up(d2)], dim=1)) 
        # out  
        flow = self.out_conv(self.up(d1))
        return flow 