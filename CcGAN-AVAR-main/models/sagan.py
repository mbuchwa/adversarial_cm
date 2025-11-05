'''

The network for Self-Attention GAN (SAGAN).

Adapted from https://github.com/voletiv/self-attention-GAN-pytorch/blob/master/sagan_models.py

'''

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils import spectral_norm
from torch.nn.init import xavier_uniform_


def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.)


def snconv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
    return spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                   stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias))

def snlinear(in_features, out_features, bias=True):
    return spectral_norm(nn.Linear(in_features=in_features, out_features=out_features, bias=bias))



class Self_Attn(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_channels):
        super(Self_Attn, self).__init__()
        self.in_channels = in_channels
        self.snconv1x1_theta = snconv2d(in_channels=in_channels, out_channels=in_channels//8, kernel_size=1, stride=1, padding=0)
        self.snconv1x1_phi = snconv2d(in_channels=in_channels, out_channels=in_channels//8, kernel_size=1, stride=1, padding=0)
        self.snconv1x1_g = snconv2d(in_channels=in_channels, out_channels=in_channels//2, kernel_size=1, stride=1, padding=0)
        self.snconv1x1_attn = snconv2d(in_channels=in_channels//2, out_channels=in_channels, kernel_size=1, stride=1, padding=0)
        self.maxpool = nn.MaxPool2d(2, stride=2, padding=0)
        self.softmax  = nn.Softmax(dim=-1)
        self.sigma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        """
            inputs :
                x : input feature maps(B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        _, ch, h, w = x.size()
        # Theta path
        theta = self.snconv1x1_theta(x)
        theta = theta.view(-1, ch//8, h*w)
        # Phi path
        phi = self.snconv1x1_phi(x)
        phi = self.maxpool(phi)
        phi = phi.view(-1, ch//8, h*w//4)
        # Attn map
        attn = torch.bmm(theta.permute(0, 2, 1), phi)
        attn = self.softmax(attn)
        # g path
        g = self.snconv1x1_g(x)
        g = self.maxpool(g)
        g = g.view(-1, ch//2, h*w//4)
        # Attn_g
        attn_g = torch.bmm(g, attn.permute(0, 2, 1))
        attn_g = attn_g.view(-1, ch//2, h, w)
        attn_g = self.snconv1x1_attn(attn_g)
        # Out
        out = x + self.sigma*attn_g
        return out




'''

Generator

'''


class ConditionalBatchNorm2d(nn.Module):
    def __init__(self, num_features, dim_embed):
        super().__init__()
        self.num_features = num_features
        self.bn = nn.BatchNorm2d(num_features, momentum=0.001, affine=False)
        self.embed_gamma = nn.Linear(dim_embed, num_features, bias=False)
        self.embed_beta = nn.Linear(dim_embed, num_features, bias=False)

    def forward(self, x, y):
        out = self.bn(x)
        gamma = self.embed_gamma(y).view(-1, self.num_features, 1, 1)
        beta = self.embed_beta(y).view(-1, self.num_features, 1, 1)
        out = out + gamma*out + beta
        return out


class GenBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dim_embed):
        super(GenBlock, self).__init__()
        self.cond_bn1 = ConditionalBatchNorm2d(in_channels, dim_embed)
        self.relu = nn.ReLU(inplace=True)
        self.snconv2d1 = snconv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.cond_bn2 = ConditionalBatchNorm2d(out_channels, dim_embed)
        self.snconv2d2 = snconv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.snconv2d0 = snconv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x, labels, upsample=True):
        x0 = x
        x = self.cond_bn1(x, labels)
        x = self.relu(x)
        if upsample:
            x = F.interpolate(x, scale_factor=2, mode='nearest') # upsample
        x = self.snconv2d1(x)
        x = self.cond_bn2(x, labels)
        x = self.relu(x)
        x = self.snconv2d2(x)
        if upsample:
            x0 = F.interpolate(x0, scale_factor=2, mode='nearest') # upsample
        x0 = self.snconv2d0(x0)
        out = x + x0
        return out


class sagan_generator(nn.Module):
    """Generator."""

    def __init__(self, dim_z, dim_y=128, nc=3, img_size=64, gene_ch=32, ch_multi=None):
        super(sagan_generator, self).__init__()

        self.dim_z = dim_z
        self.gene_ch = gene_ch
        self.img_size = img_size
        assert self.img_size in [64, 128, 192, 256]
        
        ## defaul setting for initial size and channel multipliers        
        if self.img_size == 64:
            self.init_size = 4
            if ch_multi is None:
                ch_multi=[16, 8, 8, 4, 4, 2, 1]
            assert len(ch_multi)>=7
        elif self.img_size == 128:
            self.init_size = 4
            if ch_multi is None:
                ch_multi=[16, 8, 8, 4, 4, 2, 2, 1]
            assert len(ch_multi)>=8   
        elif self.img_size == 192:
            self.init_size = 3
            if ch_multi is None:
                ch_multi=[16, 8, 8, 4, 4, 2, 2, 1, 1]
            assert len(ch_multi)>=9
        elif self.img_size == 256:
            self.init_size = 4
            if ch_multi is None:
                ch_multi=[16, 8, 8, 4, 4, 2, 2, 1, 1]
            assert len(ch_multi)>=9
        self.ch_multi = ch_multi
        
                
        self.snlinear0 = snlinear(in_features=dim_z, out_features=gene_ch*ch_multi[0]*self.init_size*self.init_size)
        self.block1 = GenBlock(gene_ch*ch_multi[0], gene_ch*ch_multi[1], dim_y) #8x8
        self.block2 = GenBlock(gene_ch*ch_multi[1], gene_ch*ch_multi[2], dim_y) #16x16
        self.block3 = GenBlock(gene_ch*ch_multi[2], gene_ch*ch_multi[3], dim_y) #32x32
        self.block4 = GenBlock(gene_ch*ch_multi[3], gene_ch*ch_multi[4], dim_y) #64x64
        
        if self.img_size in [64]:
            self.block5 = GenBlock(gene_ch*ch_multi[4], gene_ch*ch_multi[5], dim_y) #64x64
            self.block6 = GenBlock(gene_ch*ch_multi[5], gene_ch*ch_multi[-1], dim_y) #64x64
        
        if self.img_size in [128]:
            self.block5 = GenBlock(gene_ch*ch_multi[4], gene_ch*ch_multi[5], dim_y) #64--->128
            self.block6 = GenBlock(gene_ch*ch_multi[5], gene_ch*ch_multi[6], dim_y) #128--->128
            self.block7 = GenBlock(gene_ch*ch_multi[6], gene_ch*ch_multi[-1], dim_y) #128--->128
            
        if self.img_size in [192, 256]:
            self.block5 = GenBlock(gene_ch*ch_multi[4], gene_ch*ch_multi[5], dim_y) #64--->128
            self.block6 = GenBlock(gene_ch*ch_multi[5], gene_ch*ch_multi[6], dim_y) #128--->256
            self.block7 = GenBlock(gene_ch*ch_multi[6], gene_ch*ch_multi[7], dim_y) #256--->256
            self.block8 = GenBlock(gene_ch*ch_multi[7], gene_ch*ch_multi[-1], dim_y) #256--->256
        
        # self-attention module
        if self.img_size in [64,128]:
            self.self_attn = Self_Attn(gene_ch*ch_multi[3]) #after block3
        elif self.img_size in [192, 256]:
            self.self_attn = Self_Attn(gene_ch*ch_multi[4]) #after block4
        
        self.bn = nn.BatchNorm2d(gene_ch*ch_multi[-1], eps=1e-5, momentum=0.0001, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.snconv2d1 = snconv2d(in_channels=gene_ch*ch_multi[-1], out_channels=nc, kernel_size=3, stride=1, padding=1)
        self.tanh = nn.Tanh()

        # Weight init
        self.apply(init_weights)

    def forward(self, z, labels):
        # labels = self.label_encoder(labels)
        # n x dim_z
        out = self.snlinear0(z)            # self.init_size x self.init_size
        out = out.view(-1, self.gene_ch*self.ch_multi[0], self.init_size, self.init_size) 
        out = self.block1(out, labels)    # 8 x 8 or 12 x 12
        out = self.block2(out, labels)    # 16 x 16 or 24 x 24
        out = self.block3(out, labels)    # 32 x 32 or 48 x 48
        if self.img_size in [64,128]:
            out = self.self_attn(out)     # 32 x 32 or 48 x 48
        out = self.block4(out, labels)    # 64 x 64 or 96 x 96
        if self.img_size in [192, 256]:
            out = self.self_attn(out)     # 64 x 64
        if self.img_size in [64]:
            out = self.block5(out, labels, upsample=False)
            out = self.block6(out, labels, upsample=False)    
        if self.img_size in [128]:
            out = self.block5(out, labels)
            out = self.block6(out, labels, upsample=False)    
            out = self.block7(out, labels, upsample=False)    
        if self.img_size in [192, 256]:
            out = self.block5(out, labels)
            out = self.block6(out, labels)
            out = self.block7(out, labels, upsample=False)
            out = self.block8(out, labels, upsample=False)
        out = self.bn(out)
        out = self.relu(out)
        out = self.snconv2d1(out)
        out = self.tanh(out)
        return out



'''

Discriminator

'''

class DiscOptBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DiscOptBlock, self).__init__()
        self.snconv2d1 = snconv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.snconv2d2 = snconv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.downsample = nn.AvgPool2d(2)
        self.snconv2d0 = snconv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x0 = x

        x = self.snconv2d1(x)
        x = self.relu(x)
        x = self.snconv2d2(x)
        x = self.downsample(x)

        x0 = self.downsample(x0)
        x0 = self.snconv2d0(x0)

        out = x + x0
        return out


class DiscBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DiscBlock, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.snconv2d1 = snconv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.snconv2d2 = snconv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.downsample = nn.AvgPool2d(2)
        self.ch_mismatch = False
        if in_channels != out_channels:
            self.ch_mismatch = True
        self.snconv2d0 = snconv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x, downsample=True):
        x0 = x

        x = self.relu(x)
        x = self.snconv2d1(x)
        x = self.relu(x)
        x = self.snconv2d2(x)
        if downsample:
            x = self.downsample(x)

        if downsample or self.ch_mismatch:
            x0 = self.snconv2d0(x0)
            if downsample:
                x0 = self.downsample(x0)

        out = x + x0
        return out


class sagan_discriminator(nn.Module):
    """Discriminator."""

    def __init__(self, dim_y=128, nc=3, img_size=64, disc_ch=32, ch_multi=None, use_aux_reg=False, use_aux_dre=False, dre_head_arch="MLP3", p_dropout=0.5):
        super(sagan_discriminator, self).__init__()
        
        self.dim_embed = dim_y
        
        self.nc = nc
        self.img_size = img_size
        assert self.img_size in [64, 128, 192, 256]
        
        self.use_aux_reg = use_aux_reg
        self.use_aux_dre = use_aux_dre
        self.dre_head_arch = dre_head_arch
        
        ## defaul setting for initial size and channel multipliers
        if self.img_size == 64:
            self.init_size = 4
            if ch_multi is None:
                ch_multi=[1, 2, 4, 8, 16]
            assert len(ch_multi)>=5
        elif self.img_size == 128:
            self.init_size = 4
            if ch_multi is None:
                ch_multi=[1, 2, 2, 4, 8, 16]
            assert len(ch_multi)>=6    
        elif self.img_size == 192:
            self.init_size = 3
            if ch_multi is None:
                ch_multi=[1, 2, 2, 4, 8, 8, 16]
            assert len(ch_multi)>=7
        elif self.img_size == 256:
            self.init_size = 4
            if ch_multi is None:
                ch_multi=[1, 2, 4, 4, 8, 8, 16]
            assert len(ch_multi)>=7
        self.ch_multi = ch_multi
        
        self.disc_ch = disc_ch
        self.opt_block1 = DiscOptBlock(nc, disc_ch*ch_multi[0])
        self.block1 = DiscBlock(disc_ch*ch_multi[0], disc_ch*ch_multi[1])
        self.block2 = DiscBlock(disc_ch*ch_multi[1], disc_ch*ch_multi[2])
        self.block3 = DiscBlock(disc_ch*ch_multi[2], disc_ch*ch_multi[3])
        self.block4 = DiscBlock(disc_ch*ch_multi[3], disc_ch*ch_multi[4])
        
        # self-attention module and rest blocks
        if self.img_size == 64:
            self.self_attn = Self_Attn(disc_ch*ch_multi[0])
        elif self.img_size in [128]:
            self.self_attn = Self_Attn(disc_ch*ch_multi[1]) 
            self.block5 = DiscBlock(disc_ch*ch_multi[4], disc_ch*ch_multi[-1])
        elif self.img_size in [192, 256]:
            self.self_attn = Self_Attn(disc_ch*ch_multi[1]) 
            self.block5 = DiscBlock(disc_ch*ch_multi[4], disc_ch*ch_multi[5])
            self.block6 = DiscBlock(disc_ch*ch_multi[5], disc_ch*ch_multi[-1])
        
        self.relu = nn.ReLU(inplace=True)
        self.snlinear1 = snlinear(in_features=disc_ch*ch_multi[-1]*self.init_size*self.init_size, out_features=1)
        self.sn_embedding1 = snlinear(self.dim_embed, disc_ch*ch_multi[-1]*self.init_size*self.init_size, bias=False)

        if use_aux_reg:
            self.reg_linear = nn.Sequential(
                spectral_norm(nn.Linear(disc_ch*ch_multi[-1]*self.init_size*self.init_size, 128)),
                nn.ReLU(),
                spectral_norm(nn.Linear(128, 1)),
                nn.ReLU(),
            )
            
        if use_aux_dre:
            if self.dre_head_arch == "MLP5":
                self.dre_linear = nn.Sequential(
                    nn.Linear(disc_ch*self.ch_multi[-1]*self.init_size*self.init_size+self.dim_embed, 2048),
                    nn.GroupNorm(8, 2048),
                    nn.ReLU(),
                    nn.Linear(2048, 1024),
                    nn.GroupNorm(8, 1024),
                    nn.ReLU(),
                    nn.Linear(1024, 512),
                    nn.GroupNorm(8, 512),
                    nn.ReLU(),
                    nn.Linear(512, 256),
                    nn.GroupNorm(8, 256),
                    nn.ReLU(),
                    nn.Linear(256, 1),
                    nn.ReLU(),
                )
            elif self.dre_head_arch == "MLP5_dropout":
                self.dre_linear = nn.Sequential(
                    nn.Linear(disc_ch*self.ch_multi[-1]*self.init_size*self.init_size+self.dim_embed, 2048),
                    nn.GroupNorm(8, 2048),
                    nn.ReLU(),
                    nn.Dropout(p_dropout),
                    nn.Linear(2048, 1024),
                    nn.GroupNorm(8, 1024),
                    nn.ReLU(),
                    nn.Dropout(p_dropout),
                    nn.Linear(1024, 512),
                    nn.GroupNorm(8, 512),
                    nn.ReLU(),
                    nn.Dropout(p_dropout),
                    nn.Linear(512, 256),
                    nn.GroupNorm(8, 256),
                    nn.ReLU(),
                    nn.Dropout(p_dropout),
                    nn.Linear(256, 1),
                    nn.ReLU(),
                )
            elif self.dre_head_arch == "MLP3_dropout":
                self.dre_linear = nn.Sequential(               
                    nn.Linear(disc_ch*self.ch_multi[-1]*self.init_size*self.init_size+self.dim_embed, 512),
                    nn.GroupNorm(8, 512),
                    nn.ReLU(),
                    nn.Dropout(p_dropout),
                    nn.Linear(512, 256),
                    nn.GroupNorm(8, 256),
                    nn.ReLU(),
                    nn.Dropout(p_dropout),
                    nn.Linear(256, 1),
                    nn.ReLU(),
                )
            elif self.dre_head_arch == "MLP3":
                self.dre_linear = nn.Sequential(               
                    nn.Linear(disc_ch*self.ch_multi[-1]*self.init_size*self.init_size+self.dim_embed, 512),
                    nn.GroupNorm(8, 512),
                    nn.ReLU(),
                    nn.Linear(512, 256),
                    nn.GroupNorm(8, 256),
                    nn.ReLU(),
                    nn.Linear(256, 1),
                    nn.ReLU(),
                )
            else:
                raise ValueError("Not Supported DRE Branch!")
        
        # Weight init
        self.apply(init_weights)
        xavier_uniform_(self.sn_embedding1.weight)

    def forward(self, x, labels):
        
        h = self.opt_block1(x)  
        if self.img_size==64:
            h = self.self_attn(h) # 32 x 32
            h = self.block1(h)    # 16 x 16
            h = self.block2(h)    # 8 x 8
            h = self.block3(h)    # 4 x 4
            h = self.block4(h, downsample=False)    # 4 x 4
        elif self.img_size in [128]:
            h = self.block1(h) # 32 x 32
            h = self.self_attn(h) # 32 x 32
            h = self.block2(h)    # 16 x 16
            h = self.block3(h)    # 8 x 8 
            h = self.block4(h)    # 4 x 4
            h = self.block5(h, downsample=False)    # 4 x 4 or 6 x 6
        elif self.img_size in [192, 256]:
            h = self.block1(h) # 64x64 or 48 x 48
            h = self.self_attn(h) # 64x64 or 48 x 48
            h = self.block2(h)    # 32x32 or 24 x 24
            h = self.block3(h)    # 16x16 or 12 x 12
            h = self.block4(h)    # 8x8 or 6 x 6  
            h = self.block5(h)    # 4x4 or 3 x 3  
            h = self.block6(h, downsample=False)    # 3x3
        
        h = self.relu(h)              # n x disc_ch*ch_multi[-1] x self.init_size x self.init_size
        h = h.view(-1, self.disc_ch*self.ch_multi[-1]*self.init_size*self.init_size)
        
        output1 = torch.squeeze(self.snlinear1(h)) # n
        # Projection
        h_labels = self.sn_embedding1(labels)   # n x disc_ch*self.ch_multi[-1]
        proj = torch.mul(h, h_labels)          # n x disc_ch*self.ch_multi[-1]
        output2 = torch.sum(proj, dim=[1])      # n
        # Out
        adv_output = (output1 + output2).view(-1,1)              # n
        
        # auxiliary regression branch
        reg_output = None
        if self.use_aux_reg:
            reg_output = self.reg_linear(h).view(-1,1)
        
        # auxiliary dre branch
        dre_output = None
        if self.use_aux_dre:
            feat_cat = torch.cat((h, labels), -1)
            dre_output = self.dre_linear(feat_cat).view(-1,1)
        
        return {
            "h":h,
            "adv_output":adv_output,
            "reg_output":reg_output,
            "dre_output":dre_output,
        }


if __name__ == "__main__":
    def get_parameter_number(net):
        total_num = sum(p.numel() for p in net.parameters())
        trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
        return {'Total': total_num, 'Trainable': trainable_num}

    IMG_SIZE=128

    netG = sagan_generator(dim_z=256, dim_y=128, nc=3, img_size=IMG_SIZE, gene_ch=64).cuda() # parameters
    netD = sagan_discriminator(dim_y=128, nc=3, img_size=IMG_SIZE, disc_ch=48, use_aux_reg=True, use_aux_dre=True, dre_head_arch="MLP3").cuda() # parameters

    # netG = nn.DataParallel(netG)
    # netD = nn.DataParallel(netD)

    N=4
    z = torch.randn(N, 256).cuda()
    y = torch.randn(N, 128).cuda()
    x = netG(z,y)
    out = netD(x,y)
    print(x.size())
    print(out['adv_output'].size())
    print(out['reg_output'].size())
    print(out['dre_output'].size())

    print('G:', get_parameter_number(netG))
    print('D:', get_parameter_number(netD))
