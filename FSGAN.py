import torch 
import torch.nn as nn 


class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, strides=1):
        super(ConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channel, out_channel, 3, strides, 1),
            nn.LeakyReLU(0.2))

    def forward(self, x):
        return self.block(x)


class MorphNet(nn.Module):
    def __init__(self, encoder_channels=[16, 32, 32, 32, 32], 
                 decoder_channels=[32, 32, 32, 8, 8]): # default vm-1
        super(MorphNet, self).__init__()

        self.down_conv0 = ConvBlock(2, encoder_channels[0])
        self.down_conv1 = ConvBlock(encoder_channels[0], encoder_channels[1], 2)
        self.down_conv2 = ConvBlock(encoder_channels[1], encoder_channels[2], 2)
        self.down_conv3 = ConvBlock(encoder_channels[2], encoder_channels[3], 2)
        self.down_conv4 = ConvBlock(encoder_channels[3], encoder_channels[4], 2)

        self.up_sample = nn.Upsample(scale_factor=2, mode='nearest')
        self.option_conv = False

        self.up_conv0 = ConvBlock(encoder_channels[4] + encoder_channels[3], decoder_channels[0])
        self.up_conv1 = ConvBlock(decoder_channels[0] + encoder_channels[2], decoder_channels[1])
        self.up_conv2 = ConvBlock(decoder_channels[1] + encoder_channels[1], decoder_channels[2])

        self.up_conv3 = ConvBlock(decoder_channels[2], decoder_channels[3])

        self.up_conv4 = ConvBlock(decoder_channels[3] + encoder_channels[0], decoder_channels[4])

        if len(decoder_channels) == 5:
            self.option_conv = False
            self.out_conv = ConvBlock(decoder_channels[4], 3)

        if len(decoder_channels) == 6:
            self.option_conv = True
            self.up_conv5 = ConvBlock(decoder_channels[4], decoder_channels[5])
            self.out_conv = ConvBlock(decoder_channels[5], 3)
            
    def forward(self, x):
        # input
        x_in = x

        # down sample path
        x0 = self.down_conv0(x_in)
        x1 = self.down_conv1(x0)
        x2 = self.down_conv2(x1)
        x3 = self.down_conv3(x2)
        x4 = self.down_conv4(x3)

        # up sample path
        x = self.up_sample(x4)
        x = torch.cat([x, x3], dim=1)
        x = self.up_conv0(x)

        x = self.up_sample(x)
        x = torch.cat([x, x2], dim=1)
        x = self.up_conv1(x)

        x = self.up_sample(x)
        x = torch.cat([x, x1], dim=1)
        x = self.up_conv2(x)

        x = self.up_conv3(x)

        x = self.up_sample(x)
        x = torch.cat([x, x0], dim=1)
        x = self.up_conv4(x)

        if self.option_conv == False:
            y = self.out_conv(x)
        else:
            x = self.up_conv5(x)
            y = self.out_conv(x)
        
        flow = y 

        return flow 
