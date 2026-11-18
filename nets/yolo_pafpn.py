



"""
改进
    1、添加了重建模型，重建模型将使用图像和backbone的三个特征进行重建
    2、添加DAFS模块，该模块将用于连接backbone和neck结构（DAFS的代码添加在 darknet 网络中）
    3、在neck的两个卷积之间，添加了1个 evc 模块
"""

import torch
import torch.nn as nn
from nets.darknet import BaseConv, CSPDarknet, CSPLayer, DWConv
from nets.decoder_net import DecoderNet
from nets.evc_blocks import EVCBlock


class YOLOPAFPN(nn.Module):
    def __init__(self, depth=1.0, width=1.0, in_features=("dark3", "dark4", "dark5"), in_channels=[256, 512, 1024],
                 depthwise=False, act="silu"):
        super().__init__()
        Conv = DWConv if depthwise else BaseConv

        self.backbone = CSPDarknet(depth, width, depthwise=depthwise, act=act)
        self.in_features = in_features
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.lateral_conv0 = BaseConv(int(in_channels[2] * width), int(in_channels[1] * width), 1, 1, act=act)


        self.C3_p4 = CSPLayer(
            int(2 * in_channels[1] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        self.reduce_conv1 = BaseConv(int(in_channels[1] * width), int(in_channels[0] * width), 1, 1, act=act)
        self.C3_p3 = CSPLayer(
            int(2 * in_channels[0] * width),
            int(in_channels[0] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        self.bu_conv2 = Conv(int(in_channels[0] * width), int(in_channels[0] * width), 3, 2, act=act)
        self.C3_n3 = CSPLayer(
            int(2 * in_channels[0] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        self.bu_conv1 = Conv(int(in_channels[1] * width), int(in_channels[1] * width), 3, 2, act=act)
        self.C3_n4 = CSPLayer(
            int(2 * in_channels[1] * width),
            int(in_channels[2] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )




        self.evcblock = EVCBlock(
            int(in_channels[1] * width),
            int(in_channels[1] * width),
            channel_ratio=4, base_channel=16,
        )


        self.decoder_net = DecoderNet()
        self.tanh = nn.Tanh()

    def forward(self, input):
        """
        在训练前对其进行了拼接的
        在dim=0维度进行拼接，是不会造成维度变化的，按batch进行分割
        """
        if self.training:
            input, clear_x = input.split((4, 4), dim=0)



        out_features = self.backbone.forward(input)
        [feat1, feat2, feat3] = [out_features[f] for f in self.in_features]



        dehazing = self.decoder_net(input, [feat1, feat2, feat3])
        dehazing = self.tanh(dehazing)


        P5 = self.lateral_conv0(feat3)
        P5_upsample = self.upsample(P5)
        P5_upsample = torch.cat([P5_upsample, feat2], 1)
        P5_upsample = self.C3_p4(P5_upsample)
        P4 = self.reduce_conv1(P5_upsample)
        P4_upsample = self.upsample(P4)

        P4_upsample = torch.cat([P4_upsample, feat1], 1)
        P3_out = self.C3_p3(P4_upsample)
        P3_downsample = self.bu_conv2(P3_out)

        P3_downsample = torch.cat([P3_downsample, P4], 1)
        P4_out = self.C3_n3(P3_downsample)
        P4_downsample = self.bu_conv1(P4_out)




        P5 = self.evcblock(P5)

        P4_downsample = torch.cat([P4_downsample, P5], 1)
        P5_out = self.C3_n4(P4_downsample)

        return P3_out, P4_out, P5_out, dehazing
