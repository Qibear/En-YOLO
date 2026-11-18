import torch.nn as nn
from torchinfo import summary
from nets.decoder_parts import Up, Down, OutConv, DoubleConv


class DecoderNet(nn.Module):

    def __init__(self):
        super().__init__()



        self.inc = (DoubleConv(3, 16))



        self.down1 = (Down(16, 32))
        self.down2 = (Down(32, 64))
        self.down3 = (Down(64, 128))
        self.down4 = (Down(128, 256))
        self.down5 = (Down(256, 512))





        self.up1 = (Up(512, 256, bilinear=False))
        self.up2 = (Up(256, 128, bilinear=False))
        self.up3 = (Up(128, 64, bilinear=False))
        self.up4 = (Up(64, 32, bilinear=False))
        self.up5 = (Up(32, 16, bilinear=False))


        self.outc = (OutConv(16, 3))

    def forward(self, x, feather_list):
        xx = self.inc(x)
        x1 = self.down1(xx)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x5 = self.down5(x4)


        x5 = x5 + feather_list[2]
        x = self.up1(x5, x4)

        x = x + feather_list[1]
        x = self.up2(x, x3)

        x = x + feather_list[0]
        x = self.up3(x, x2)

        x = self.up4(x, x1)
        x = self.up5(x, xx)
        out = self.outc(x)
        return out


if __name__ == '__main__':
    model = DecoderNet()
    summary(model, (4, 3, 640, 640))
