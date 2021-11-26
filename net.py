import torch
import torch.nn as nn
from utils import mean_variance_norm, DEVICE
from utils import calc_ss_loss, calc_remd_loss, calc_moment_loss, calc_mse_loss, calc_histogram_loss
from hist_loss import RGBuvHistBlock
import torch

class Net(nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()
        self.args = args
        self.vgg = vgg19[:44]
        self.vgg.load_state_dict(torch.load('./checkpoints/encoder.pth', map_location='cpu'), strict=False)
        for param in self.vgg.parameters():
            param.requires_grad = False

        self.align1 = PAMA(512)
        self.align2 = PAMA(512)
        self.align3 = PAMA(512)

        self.decoder = decoder
        self.hist = RGBuvHistBlock(insz=64, h=256, 
                                   intensity_scale=True, 
                                   method='inverse-quadratic',
                                   device=DEVICE)

        if args.pretrained == True:
            self.align1.load_state_dict(torch.load('./checkpoints/PAMA1.pth', map_location='cpu'), strict=True)
            self.align2.load_state_dict(torch.load('./checkpoints/PAMA2.pth', map_location='cpu'), strict=True)
            self.align3.load_state_dict(torch.load('./checkpoints/PAMA3.pth', map_location='cpu'), strict=True)
            self.decoder.load_state_dict(torch.load('./checkpoints/decoder.pth', map_location='cpu'), strict=False)

        if args.requires_grad == False:
            for param in self.parameters():
                param.requires_grad = False


    def forward(self, Ic, Is):
        feat_c = self.forward_vgg(Ic)
        feat_s = self.forward_vgg(Is)
        Fc, Fs = feat_c[3], feat_s[3]

        Fcs1 = self.align1(Fc, Fs)
        Fcs2 = self.align2(Fcs1, Fs)
        Fcs3 = self.align3(Fcs2, Fs)

        Ics3 = self.decoder(Fcs3)

        if self.args.training == True:
            Ics1 = self.decoder(Fcs1)
            Ics2 = self.decoder(Fcs2)
            Irc = self.decoder(Fc)
            Irs = self.decoder(Fs)
            feat_cs1 = self.forward_vgg(Ics1)
            feat_cs2 = self.forward_vgg(Ics2)
            feat_cs3 = self.forward_vgg(Ics3)
            feat_rc = self.forward_vgg(Irc)
            feat_rs = self.forward_vgg(Irs)

            content_loss1, remd_loss1, moment_loss1, color_loss1 = 0.0, 0.0, 0.0, 0.0
            content_loss2, remd_loss2, moment_loss2, color_loss2 = 0.0, 0.0, 0.0, 0.0
            content_loss3, remd_loss3, moment_loss3, color_loss3 = 0.0, 0.0, 0.0, 0.0
            loss_rec = 0.0

            for l in range(2, 5):     
                content_loss1 += self.args.w_content1 * calc_ss_loss(feat_cs1[l], feat_c[l])
                remd_loss1 += self.args.w_remd1 * calc_remd_loss(feat_cs1[l], feat_s[l])
                moment_loss1 += self.args.w_moment1 * calc_moment_loss(feat_cs1[l], feat_s[l])

                content_loss2 += self.args.w_content2 * calc_ss_loss(feat_cs2[l], feat_c[l])
                remd_loss2 += self.args.w_remd2 * calc_remd_loss(feat_cs2[l], feat_s[l])
                moment_loss2 += self.args.w_moment2 * calc_moment_loss(feat_cs2[l], feat_s[l])

                content_loss3 += self.args.w_content3 * calc_ss_loss(feat_cs3[l], feat_c[l])
                remd_loss3 += self.args.w_remd3 * calc_remd_loss(feat_cs3[l], feat_s[l])
                moment_loss3 += self.args.w_moment3 * calc_moment_loss(feat_cs3[l], feat_s[l])

                loss_rec += 0.5 * calc_mse_loss(feat_rc[l], feat_c[l]) + 0.5 * calc_mse_loss(feat_rs[l], feat_s[l])
            loss_rec += 25 * calc_mse_loss(Irc, Ic)
            loss_rec += 25 * calc_mse_loss(Irs, Is)

            if self.args.color_on:
                color_loss1 += self.args.w_color1 * calc_histogram_loss(Ics1, Is, self.hist)
                color_loss2 += self.args.w_color2 * calc_histogram_loss(Ics2, Is, self.hist)
                color_loss3 += self.args.w_color3 * calc_histogram_loss(Ics3, Is, self.hist)
            
            loss1 = (content_loss1+remd_loss1+moment_loss1+color_loss1)/(self.args.w_content1+self.args.w_remd1+self.args.w_moment1+self.args.w_color1)
            loss2 = (content_loss2+remd_loss2+moment_loss2+color_loss2)/(self.args.w_content2+self.args.w_remd2+self.args.w_moment2+self.args.w_color2)
            loss3 = (content_loss3+remd_loss3+moment_loss3+color_loss3)/(self.args.w_content3+self.args.w_remd3+self.args.w_moment3+self.args.w_color3)
            loss = loss1 + loss2 + loss3 + loss_rec
            return loss
        else: 
            return Ics3

    def forward_vgg(self, x):
        relu1_1 = self.vgg[:4](x)
        relu2_1 = self.vgg[4:11](relu1_1)
        relu3_1 = self.vgg[11:18](relu2_1)
        relu4_1 = self.vgg[18:31](relu3_1)
        relu5_1 = self.vgg[31:44](relu4_1)
        return [relu1_1, relu2_1, relu3_1, relu4_1, relu5_1]
    
    def save_ckpts(self):
        torch.save(self.align1.state_dict(), "./checkpoints/PAMA1.pth")
        torch.save(self.align2.state_dict(), "./checkpoints/PAMA2.pth")
        torch.save(self.align3.state_dict(), "./checkpoints/PAMA3.pth")
        torch.save(self.decoder.state_dict(), "./checkpoints/decoder.pth")  

#---------------------------------------------------------------------------------------------------------------

vgg19 = nn.Sequential(
    nn.Conv2d(3, 3, (1, 1)),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(3, 64, (3, 3)),
    nn.ReLU(),  # relu1-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),  # relu1-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 128, (3, 3)),
    nn.ReLU(),  # relu2-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),  # relu2-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 256, (3, 3)),
    nn.ReLU(),  # relu3-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 512, (3, 3)),
    nn.ReLU(),  # relu4-1, 
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU()  # relu5-4
)

#---------------------------------------------------------------------------------------------------------------

decoder = nn.Sequential(
    nn.ReflectionPad2d((1, 1, 1, 1)),  
    nn.Conv2d(512, 256, (3, 3)),
    nn.ReLU(),  #relu4_1
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 128, (3, 3)),
    nn.ReLU(),  #relu3_1
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 64, (3, 3)),
    nn.ReLU(),  #relu2_1
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),  #relu1_1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 3, (3, 3)),
)

#---------------------------------------------------------------------------------------------------------------

class AttentionUnit(nn.Module):
    def __init__(self, channels):
        super(AttentionUnit, self).__init__()
        self.relu6 = nn.ReLU6()
        self.f = nn.Conv2d(channels, channels//2, (1, 1))
        self.g = nn.Conv2d(channels, channels//2, (1, 1))
        self.h = nn.Conv2d(channels, channels//2, (1, 1))

        self.out_conv = nn.Conv2d(channels//2, channels, (1, 1))
        self.softmax = nn.Softmax(dim = -1)
        
    def forward(self, Fc, Fs):
        B, C, H, W = Fc.shape
        f_Fc = self.relu6(self.f(mean_variance_norm(Fc)))
        g_Fs = self.relu6(self.g(mean_variance_norm(Fs)))
        h_Fs = self.relu6(self.h(Fs))
        f_Fc = f_Fc.view(f_Fc.shape[0], f_Fc.shape[1], -1).permute(0, 2, 1)
        g_Fs = g_Fs.view(g_Fs.shape[0], g_Fs.shape[1], -1)

        Attention = self.softmax(torch.bmm(f_Fc, g_Fs))

        h_Fs = h_Fs.view(h_Fs.shape[0], h_Fs.shape[1], -1)

        Fcs = torch.bmm(h_Fs, Attention.permute(0, 2, 1))
        Fcs = Fcs.view(B, C//2, H, W)
        Fcs = self.relu6(self.out_conv(Fcs))

        return Fcs

class FuseUnit(nn.Module):
    def __init__(self, channels):
        super(FuseUnit, self).__init__()
        self.proj1 = nn.Conv2d(2*channels, channels, (1, 1))
        self.proj2 = nn.Conv2d(channels, channels, (1, 1))
        self.proj3 = nn.Conv2d(channels, channels, (1, 1))

        self.fuse1x = nn.Conv2d(channels, 1, (1, 1), stride = 1)
        self.fuse3x = nn.Conv2d(channels, 1, (3, 3), stride = 1)
        self.fuse5x = nn.Conv2d(channels, 1, (5, 5), stride = 1)

        self.pad3x = nn.ReflectionPad2d((1, 1, 1, 1))
        self.pad5x = nn.ReflectionPad2d((2, 2, 2, 2))
        self.sigmoid = nn.Sigmoid()

    def forward(self, F1, F2):
        Fcat = self.proj1(torch.cat((F1, F2), dim=1))
        F1 = self.proj2(F1)
        F2 = self.proj3(F2)
        
        fusion1 = self.sigmoid(self.fuse1x(Fcat))      
        fusion3 = self.sigmoid(self.fuse3x(self.pad3x(Fcat)))
        fusion5 = self.sigmoid(self.fuse5x(self.pad5x(Fcat)))
        fusion = (fusion1 + fusion3 + fusion5) / 3

        return torch.clamp(fusion, min=0, max=1.0)*F1 + torch.clamp(1 - fusion, min=0, max=1.0)*F2 
        
class PAMA(nn.Module):
    def __init__(self, channels):
        super(PAMA, self).__init__()
        self.conv_in = nn.Conv2d(channels, channels, (3, 3), stride=1)
        self.attn = AttentionUnit(channels)
        self.fuse = FuseUnit(channels)
        self.conv_out = nn.Conv2d(channels, channels, (3, 3), stride=1)

        self.pad = nn.ReflectionPad2d((1, 1, 1, 1))
        self.relu6 = nn.ReLU6()
    
    def forward(self, Fc, Fs):
        Fc = self.relu6(self.conv_in(self.pad(Fc)))
        Fs = self.relu6(self.conv_in(self.pad(Fs)))
        Fcs = self.attn(Fc, Fs)
        Fcs = self.relu6(self.conv_out(self.pad(Fcs)))
        Fcs = self.fuse(Fc, Fcs)
        
        return Fcs
    
#---------------------------------------------------------------------------------------------------------------


