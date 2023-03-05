import mindspore 
import mindspore.nn as nn 
from .blocks import ConvBlock, TransformerBlock, TransformerBlockSingle


class MainNet(nn.Cell):
    def __init__(self, enc_nc=[16, 32, 32], dec_nc=[32, 32, 32, 16], patch_size=4, num_heads=4):
        super().__init__()
        # enc conv 
        self.conv_enc11 = ConvBlock(2, enc_nc[0], 3, 1)
        self.conv_enc12 = ConvBlock(enc_nc[0], enc_nc[0], 3, 1)

        self.conv_enc21 = ConvBlock(enc_nc[0], enc_nc[1], 3, 1)
        self.tf_enc21 = TransformerBlock(enc_nc[1], patch_size, num_heads, 'down')
        self.conv_enc22 = ConvBlock(enc_nc[1], enc_nc[1], 3, 1)

        self.conv_enc31 = ConvBlock(enc_nc[1] + enc_nc[1], enc_nc[2], 3, 1)
        self.tf_enc31 = TransformerBlock(enc_nc[2], patch_size, num_heads, 'down')
        self.conv_enc32 = ConvBlock(enc_nc[2], enc_nc[2], 3, 1)
        # dec conv 
        self.conv_dec41 = ConvBlock(enc_nc[2] + enc_nc[2], dec_nc[0], 3, 1)
        self.tf_dec41 = TransformerBlock(dec_nc[0], patch_size, num_heads, 'up')
        self.conv_dec42 = ConvBlock(dec_nc[0], dec_nc[0], 3, 1)

        self.pool = nn.MaxPool3d(kernel_size=3, padding=1, stride=2)
        self.up = nn.Upsample(scale_factor=2, mode='nearest')

        self.conv_dec31 = ConvBlock(enc_nc[2] + dec_nc[0] + dec_nc[0], dec_nc[1], 3, 1)
        self.tf_dec31 = TransformerBlock(dec_nc[1], patch_size, num_heads, 'up')
        self.conv_dec32 = ConvBlock(dec_nc[1], dec_nc[1], 3, 1)

        self.conv_dec21 = ConvBlock(enc_nc[1] + dec_nc[1] + dec_nc[1], dec_nc[2], 3, 1)
        self.tf_dec21 = TransformerBlock(dec_nc[2], patch_size, num_heads, 'up')
        self.conv_dec22 = ConvBlock(dec_nc[2], dec_nc[2], 3, 1)

        self.conv_dec11 = ConvBlock(enc_nc[0] + dec_nc[2] + dec_nc[2], dec_nc[3], 3, 1)
        self.conv_dec12 = ConvBlock(dec_nc[3], 3, 3, 1)

    def construct(self, x):
        x11 = self.conv_enc11(x)
        x12 = self.conv_enc12(x11)
        x20 = self.pool(x12)

        x21 = self.conv_enc21(x20)
        s21, d21 = self.tf_enc21(x21) # s: same d: down u:up
        x22 = self.conv_enc22(s21)
        x30 = self.pool(x22)

        x31 = self.conv_enc31(mindspore.cat([d21, x30], dim=1))
        s31, d31 = self.tf_enc31(x31)
        x32 = self.conv_enc32(s31)
        x40 = self.pool(x32)

        x41 = self.conv_dec41(mindspore.cat([d31, x40], dim=1))
        s41, u41 = self.tf_dec41(x41)
        x42 = self.conv_dec42(s41)
        x42 = self.up(x42)

        xx31 = self.conv_dec31(mindspore.cat([x32, u41, x42], dim=1))
        ss31, u31 = self.tf_dec31(xx31)
        xx32 = self.conv_dec32(ss31)
        xx32 = self.up(xx32)

        xx21 = self.conv_dec21(mindspore.cat([x22, u31, xx32], dim=1))
        ss21, u21 = self.tf_dec21(xx21)
        xx22 = self.conv_dec22(ss21)
        xx22 = self.up(xx22)

        xx11 = self.conv_dec11(mindspore.cat([x12, u21, xx22], dim=1))
        flow = self.conv_dec12(xx11)

        return flow 


class MainNetWithoutTF(nn.Cell): 
    def __init__(self, enc_nc=[16, 32, 32], dec_nc=[32, 32, 32, 16], patch_size=4, num_heads=4):
        super().__init__()
        # enc conv 
        self.conv_enc11 = ConvBlock(2, enc_nc[0], 3, 1)
        self.conv_enc12 = ConvBlock(enc_nc[0], enc_nc[0], 3, 1)

        self.conv_enc21 = ConvBlock(enc_nc[0], enc_nc[1], 3, 1)
        self.conv_enc22 = ConvBlock(enc_nc[1], enc_nc[1], 3, 1)

        self.conv_enc31 = ConvBlock(enc_nc[1], enc_nc[2], 3, 1)
        self.conv_enc32 = ConvBlock(enc_nc[2], enc_nc[2], 3, 1)
        
        # dec conv 
        self.conv_dec41 = ConvBlock(enc_nc[2], dec_nc[0], 3, 1)
        self.conv_dec42 = ConvBlock(dec_nc[0], dec_nc[0], 3, 1)

        self.pool = nn.MaxPool3d(kernel_size=3, padding=1, stride=2)
        self.up = nn.Upsample(scale_factor=2, mode='nearest')

        self.conv_dec31 = ConvBlock(enc_nc[2] + dec_nc[0], dec_nc[1], 3, 1)
        self.conv_dec32 = ConvBlock(dec_nc[1], dec_nc[1], 3, 1)

        self.conv_dec21 = ConvBlock(enc_nc[1] + dec_nc[1], dec_nc[2], 3, 1)
        self.conv_dec22 = ConvBlock(dec_nc[2], dec_nc[2], 3, 1)

        self.conv_dec11 = ConvBlock(enc_nc[0] + dec_nc[2], dec_nc[3], 3, 1)
        self.conv_dec12 = ConvBlock(dec_nc[3], 3, 3, 1)

    def construct(self, x):
        x11 = self.conv_enc11(x)
        x12 = self.conv_enc12(x11)
        x20 = self.pool(x12)

        x21 = self.conv_enc21(x20)
        x22 = self.conv_enc22(x21)
        x30 = self.pool(x22)

        x31 = self.conv_enc31(x30)
        x32 = self.conv_enc32(x31)
        x40 = self.pool(x32)

        x41 = self.conv_dec41(x40)
        x42 = self.conv_dec42(x41)
        x42 = self.up(x42)

        xx31 = self.conv_dec31(mindspore.cat([x32, x42], dim=1))
        xx32 = self.conv_dec32(xx31)
        xx32 = self.up(xx32)

        xx21 = self.conv_dec21(mindspore.cat([x22, xx32], dim=1))
        xx22 = self.conv_dec22(xx21)
        xx22 = self.up(xx22)

        xx11 = self.conv_dec11(mindspore.cat([x12, xx22], dim=1))
        flow = self.conv_dec12(xx11)

        return flow 
    

class MainNetWithSingleTF(nn.Module):
    def __init__(self, enc_nc=[16, 32, 32], dec_nc=[32, 32, 32, 16], patch_size=4, num_heads=4):
        super().__init__()
        # enc conv 
        self.conv_enc11 = ConvBlock(2, enc_nc[0], 3, 1)
        self.conv_enc12 = ConvBlock(enc_nc[0], enc_nc[0], 3, 1)

        self.conv_enc21 = ConvBlock(enc_nc[0], enc_nc[1], 3, 1)
        self.tf_enc21 = TransformerBlockSingle(enc_nc[1], patch_size, num_heads)
        self.conv_enc22 = ConvBlock(enc_nc[1], enc_nc[1], 3, 1)

        self.conv_enc31 = ConvBlock(enc_nc[1], enc_nc[2], 3, 1)
        self.tf_enc31 = TransformerBlockSingle(enc_nc[2], patch_size, num_heads)
        self.conv_enc32 = ConvBlock(enc_nc[2], enc_nc[2], 3, 1)
        # dec conv 
        self.conv_dec41 = ConvBlock(enc_nc[2], dec_nc[0], 3, 1)
        self.tf_dec41 = TransformerBlockSingle(dec_nc[0], patch_size, num_heads)
        self.conv_dec42 = ConvBlock(dec_nc[0], dec_nc[0], 3, 1)

        self.pool = nn.MaxPool3d(kernel_size=3, padding=1, stride=2)
        self.up = nn.Upsample(scale_factor=2, mode='nearest')

        self.conv_dec31 = ConvBlock(enc_nc[2] + dec_nc[0], dec_nc[1], 3, 1)
        self.tf_dec31 = TransformerBlockSingle(dec_nc[1], patch_size, num_heads)
        self.conv_dec32 = ConvBlock(dec_nc[1], dec_nc[1], 3, 1)

        self.conv_dec21 = ConvBlock(enc_nc[1] + dec_nc[1], dec_nc[2], 3, 1)
        self.tf_dec21 = TransformerBlockSingle(dec_nc[2], patch_size, num_heads)
        self.conv_dec22 = ConvBlock(dec_nc[2], dec_nc[2], 3, 1)

        self.conv_dec11 = ConvBlock(enc_nc[0] + dec_nc[2], dec_nc[3], 3, 1)
        self.conv_dec12 = ConvBlock(dec_nc[3], 3, 3, 1)

    def construct(self, x):
        x11 = self.conv_enc11(x)
        x12 = self.conv_enc12(x11)
        x20 = self.pool(x12)

        x21 = self.conv_enc21(x20)
        s21 = self.tf_enc21(x21) # s: same d: down u:up
        x22 = self.conv_enc22(s21)
        x30 = self.pool(x22)

        x31 = self.conv_enc31(x30)
        s31 = self.tf_enc31(x31)
        x32 = self.conv_enc32(s31)
        x40 = self.pool(x32)

        x41 = self.conv_dec41(x40)
        s41 = self.tf_dec41(x41)
        x42 = self.conv_dec42(s41)
        x42 = self.up(x42)

        xx31 = self.conv_dec31(mindspore.cat([x32, x42], dim=1))
        ss31 = self.tf_dec31(xx31)
        xx32 = self.conv_dec32(ss31)
        xx32 = self.up(xx32)

        xx21 = self.conv_dec21(mindspore.cat([x22, xx32], dim=1))
        ss21 = self.tf_dec21(xx21)
        xx22 = self.conv_dec22(ss21)
        xx22 = self.up(xx22)

        xx11 = self.conv_dec11(mindspore.cat([x12, xx22], dim=1))
        flow = self.conv_dec12(xx11)

        return flow 
