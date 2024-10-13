import torch.nn as nn
import torch.nn.functional as F
from .MobileNetV2 import mobilenet_v2
import  torch


class FFM(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(FFM, self).__init__()
        self.conv1=nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0)
        self.relu = nn.ReLU(inplace=True)

    def forward(self,x1,x2):
        x1=self.conv1(x1)
        x1=self.relu(x1)
        x2 = self.conv1(x2)
        x_x=x1*x2
        x_x=self.relu(x_x)
        x_x=x_x+x2
        x_x=x_x*x1
        x_x=self.relu(x_x)
        return x_x


class GMM(nn.Module):

    def __init__(self, num_channels, epsilon=1e-5, mode='l2', after_relu=False):
        super(GMM, self).__init__()

        self.alpha = nn.Parameter(torch.ones(1, num_channels, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.epsilon = epsilon
        self.mode = mode
        self.after_relu = after_relu

    def forward(self, x):
        embedding = (x.pow(2).sum((2, 3), keepdim=True) +self.epsilon).pow(0.5) * self.alpha
        norm = self.gamma / (embedding.pow(2).mean(dim=1, keepdim=True) + self.epsilon).pow(0.5)
        gate = 1. + torch.tanh(embedding * norm + self.beta)
        return x * gate


class ChannelExchange(nn.Module):
    def __init__(self, p=2):
        super().__init__()
        self.p = p
    def forward(self, x1, x2):
        N, C, H, W = x1.shape
        exchange_mask = torch.arange(C) % self.p == 0
        exchange_mask = exchange_mask.unsqueeze(0).expand((N, -1))
        out_x1, out_x2 = torch.zeros_like(x1), torch.zeros_like(x2)
        out_x1[~exchange_mask, ...] = x1[~exchange_mask, ...]
        out_x2[~exchange_mask, ...] = x2[~exchange_mask, ...]
        out_x1[exchange_mask, ...] = x2[exchange_mask, ...]
        out_x2[exchange_mask, ...] = x1[exchange_mask, ...]
        return out_x1, out_x2


class SqueezeDoubleConvOld(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SqueezeDoubleConvOld, self).__init__()
        self.squeeze = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.GELU())
        self.double_conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),
            )
        self.acfun = nn.GELU()
        self.gmm = GMM(out_channels)

    def forward(self, x):
        x = self.squeeze(x)
        x=self.gmm(x)
        block_x = self.double_conv(x)
        x = self.acfun(x + block_x)
        return  x



class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.gtc=GMM(in_planes)
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x=self.gtc(x)
        #print(x.shape)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class LCD_Net(nn.Module):
    def __init__(self,):
        super(LCD_Net, self).__init__()

        mob = mobilenet_v2(pretrained=True)

        self.inc = mob.features[:2]     #16
        self.down1 = mob.features[2:4]  # 24
        self.down2 = mob.features[4:7]  # 32
        self.down3 = mob.features[7:14]  # 96
        self.down4 = mob.features[14:18]  # 320
        self.cont1 = SqueezeDoubleConvOld(24,64)
        self.cont2 = SqueezeDoubleConvOld(32, 64)
        self.cont3 = SqueezeDoubleConvOld(96, 64)
        self.cont4 = SqueezeDoubleConvOld(320, 64)


        self.decoder = nn.Sequential(SqueezeDoubleConvOld(472, 64),nn.Conv2d(64,1,1))

        self.decoder_4 = nn.Sequential(SqueezeDoubleConvOld(64*2+1, 64))
        self.decoder_3 = nn.Sequential(SqueezeDoubleConvOld(64 * 3+1, 64))
        self.decoder_2 = nn.Sequential(SqueezeDoubleConvOld(64 * 3+1, 64))
        self.decoder_1 = nn.Sequential(SqueezeDoubleConvOld(64 * 3+1, 64))
        self.decoder_final = nn.Sequential(SqueezeDoubleConvOld(64,64),nn.Conv2d(64,1,1))
        self.chan=ChannelExchange()

        self.ffm = FFM(472 ,472 )

    def forward(self,A,B):
        size = A.size()[2:]
        layer1_pre = self.inc(A)
        layer2_pre = self.inc(B)
        layer1_A = self.down1(layer1_pre)
        layer1_B = self.down1(layer2_pre)
        layer2_A = self.down2(layer1_A)
        layer2_B = self.down2(layer1_B)
        layer2_A, layer2_B = self.chan(layer2_A, layer2_B)
        layer3_A = self.down3(layer2_A)
        layer3_B = self.down3(layer2_B)
        layer3_A, layer3_B = self.chan(layer3_A, layer3_B)
        layer4_A = self.down4(layer3_A)
        layer4_B = self.down4(layer3_B)
        layer4_A, layer4_B = self.chan(layer4_A, layer4_B)
        layer4_As = F.interpolate(layer4_A, layer1_A.size()[2:], mode='bilinear', align_corners=True)
        layer3_As = F.interpolate(layer3_A, layer1_A.size()[2:], mode='bilinear', align_corners=True)
        layer2_As = F.interpolate(layer2_A, layer1_A.size()[2:], mode='bilinear', align_corners=True)
        layer1_As = F.interpolate(layer1_A, layer1_A.size()[2:], mode='bilinear', align_corners=True)
        layer4_Bs = F.interpolate(layer4_B, layer1_A.size()[2:], mode='bilinear', align_corners=True)
        layer3_Bs = F.interpolate(layer3_B, layer1_A.size()[2:], mode='bilinear', align_corners=True)
        layer2_Bs = F.interpolate(layer2_B, layer1_A.size()[2:], mode='bilinear', align_corners=True)
        layer1_Bs = F.interpolate(layer1_B, layer1_A.size()[2:], mode='bilinear', align_corners=True)


        layer_As1=torch.cat([layer1_As,layer2_As,layer3_As,layer4_As],dim=1)
        layer_Bs1=torch.cat([layer1_Bs,layer2_Bs,layer3_Bs,layer4_Bs],dim=1)
        layer_ss=self.ffm(layer_As1,layer_Bs1)
        layer1_A=self.cont1(layer1_A)
        layer2_A=self.cont2(layer2_A)
        layer3_A=self.cont3(layer3_A)
        layer4_A=self.cont4(layer4_A)
        layer1_B=self.cont1(layer1_B)
        layer2_B=self.cont2(layer2_B)
        layer3_B=self.cont3(layer3_B)
        layer4_B=self.cont4(layer4_B)

        layer1 = torch.cat((layer1_B,layer1_A),dim=1)
        layer2 = torch.cat((layer2_B,layer2_A),dim=1)
        layer3 = torch.cat((layer3_B,layer3_A),dim=1)
        layer4 = torch.cat((layer4_B, layer4_A), dim=1)
        layer4_1 = F.interpolate(layer4, layer1.size()[2:], mode='bilinear', align_corners=True)
        layer3_1 = F.interpolate(layer3, layer1.size()[2:], mode='bilinear', align_corners=True)
        layer2_1 = F.interpolate(layer2, layer1.size()[2:], mode='bilinear', align_corners=True)
        layer1_1 = F.interpolate(layer1, layer1.size()[2:], mode='bilinear', align_corners=True)

        feature_fuse = layer_ss
        change_map = self.decoder(feature_fuse)
        change_map = F.interpolate(change_map, size, mode='bilinear', align_corners=True)
        change_map1 = F.interpolate(change_map, layer1.size()[2:], mode='bilinear', align_corners=True)

        layer4_1 = torch.cat([layer4_1, change_map1], dim=1)
        layer4_1 = self.decoder_4(layer4_1)
        layer3_1 = torch.cat([layer4_1, layer3_1, change_map1], dim=1)
        layer3_1 = self.decoder_3(layer3_1)
        layer2_1 = torch.cat([layer3_1, layer2_1, change_map1], dim=1)
        layer2_1 = self.decoder_2(layer2_1)
        layer1_1 = torch.cat([layer2_1, layer1_1, change_map1], dim=1)
        layer1_1 = self.decoder_1(layer1_1)
        final_map = self.decoder_final(layer1_1)
        # print(final_map.shape,'fina')
        final_map = F.interpolate(final_map, size, mode='bilinear', align_corners=True)
        return change_map, final_map

if __name__ == '__main__':
    model = LCD_Net()
    img = torch.randn(1, 3, 256, 256)
    img1 = torch.randn(1, 3, 256, 256)
    res = model(img, img1)
    print(res[0].shape)

    from thop import profile
   # mmengine_flop_count(model, (3, 512, 512), show_table=True, show_arch=True)
    flops1, params1 = profile(model, inputs=(img,img1))
    print("flops=G", flops1 )
    print("parms=M", params1 )