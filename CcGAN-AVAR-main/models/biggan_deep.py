'''
The network for BigGAN-deep

https://github.com/POSTECH-CVLab/PyTorch-StudioGAN

https://github.com/ajbrock/BigGAN-PyTorch

https://github.com/voletiv/self-attention-GAN-pytorch/blob/master/sagan_models.py

'''

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils import spectral_norm
from torch.nn.init import xavier_uniform_
from torch.nn import init

def myconv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, use_sn=False):
    if use_sn: # apply spectral normalization
        return spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias))
    else:
        return nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)

def mylinear(in_features, out_features, bias=True, use_sn=False):
    if use_sn:
        return spectral_norm(nn.Linear(in_features=in_features, out_features=out_features, bias=bias))
    else:
        return nn.Linear(in_features=in_features, out_features=out_features, bias=bias)

def batchnorm_2d(in_features, eps=1e-4, momentum=0.1, affine=True):
    return nn.BatchNorm2d(in_features, eps=eps, momentum=momentum, affine=affine, track_running_stats=True)


class Self_Attn(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_channels, use_sn):
        super(Self_Attn, self).__init__()
        self.in_channels = in_channels
        self.snconv1x1_theta = myconv2d(in_channels=in_channels, out_channels=in_channels//8, kernel_size=1, stride=1, padding=0, use_sn=use_sn)
        self.snconv1x1_phi = myconv2d(in_channels=in_channels, out_channels=in_channels//8, kernel_size=1, stride=1, padding=0, use_sn=use_sn)
        self.snconv1x1_g = myconv2d(in_channels=in_channels, out_channels=in_channels//2, kernel_size=1, stride=1, padding=0, use_sn=use_sn)
        self.snconv1x1_attn = myconv2d(in_channels=in_channels//2, out_channels=in_channels, kernel_size=1, stride=1, padding=0, use_sn=use_sn)
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


def init_weights(modules, initialize):
    for module in modules():
        if (isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d) or isinstance(module, nn.Linear)):
            if initialize == "ortho":
                init.orthogonal_(module.weight)
                if module.bias is not None:
                    module.bias.data.fill_(0.)
            elif initialize == "N02":
                init.normal_(module.weight, 0, 0.02)
                if module.bias is not None:
                    module.bias.data.fill_(0.)
            elif initialize in ["glorot", "xavier"]:
                init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    module.bias.data.fill_(0.)
            else:
                pass
        elif isinstance(module, nn.Embedding):
            if initialize == "ortho":
                init.orthogonal_(module.weight)
            elif initialize == "N02":
                init.normal_(module.weight, 0, 0.02)
            elif initialize in ["glorot", "xavier"]:
                init.xavier_uniform_(module.weight)
            else:
                pass
        else:
            pass




######################################################
# generator

class ConditionalBatchNorm2d(nn.Module):
    def __init__(self, num_features, dim_cond, use_sn=False):
        super().__init__()
        self.num_features = num_features
        self.bn = nn.BatchNorm2d(num_features, eps=1e-4, momentum=0.1, affine=False)
        self.embed_gamma = mylinear(dim_cond, num_features, bias=False, use_sn=use_sn)
        self.embed_beta = mylinear(dim_cond, num_features, bias=False, use_sn=use_sn)

    def forward(self, x, y):
        out = self.bn(x)
        gamma = self.embed_gamma(y).view(-1, self.num_features, 1, 1)
        beta = self.embed_beta(y).view(-1, self.num_features, 1, 1)
        out = out + gamma*out + beta
        return out
    

class GenBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dim_cond, use_sn=False, channel_ratio=4, upsample=True):
        super(GenBlock, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dim_cond = dim_cond
        self.use_sn = use_sn
        self.upsample = upsample
        
        self.hidden_channels = self.in_channels // channel_ratio

        self.cond_bn1 = ConditionalBatchNorm2d(self.in_channels, dim_cond)
        self.cond_bn2 = ConditionalBatchNorm2d(self.hidden_channels, dim_cond)
        self.cond_bn3 = ConditionalBatchNorm2d(self.hidden_channels, dim_cond)
        self.cond_bn4 = ConditionalBatchNorm2d(self.hidden_channels, dim_cond)

        self.relu = nn.ReLU(inplace=True)
        
        self.conv2d0 = myconv2d(in_channels=self.in_channels,
                                out_channels=self.out_channels,
                                kernel_size=1,
                                stride=1,
                                padding=0, 
                                use_sn=use_sn)
        self.conv2d1 = myconv2d(in_channels=self.in_channels,
                                out_channels=self.hidden_channels,
                                kernel_size=1,
                                stride=1,
                                padding=0, 
                                use_sn=use_sn)
        self.conv2d2 = myconv2d(in_channels=self.hidden_channels,
                                out_channels=self.hidden_channels,
                                kernel_size=3,
                                stride=1,
                                padding=1, 
                                use_sn=use_sn)
        self.conv2d3 = myconv2d(in_channels=self.hidden_channels,
                                out_channels=self.hidden_channels,
                                kernel_size=3,
                                stride=1,
                                padding=1, 
                                use_sn=use_sn)
        self.conv2d4 = myconv2d(in_channels=self.hidden_channels,
                                out_channels=self.out_channels,
                                kernel_size=1,
                                stride=1,
                                padding=0, 
                                use_sn=use_sn)
        
    
    def forward(self, x, zy):
        
        x0 = x
        
        x = self.cond_bn1(x, zy)
        x = self.relu(x)
        x = self.conv2d1(x)
        
        x = self.cond_bn2(x, zy)
        x = self.relu(x)
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode="nearest")  # upsample
        x = self.conv2d2(x)

        x = self.cond_bn3(x, zy)
        x = self.relu(x)
        x = self.conv2d3(x)
        
        x = self.cond_bn4(x, zy)
        x = self.relu(x)
        x = self.conv2d4(x)

        if self.upsample:
            x0 = F.interpolate(x0, scale_factor=2, mode="nearest")  # upsample
        x0 = self.conv2d0(x0)
        out = x + x0
        
        return out


class biggan_deep_generator(nn.Module):
    def __init__(self, dim_z, dim_y, img_size, nc, gene_ch, ch_multi=None, use_sn=True, use_attn=True, g_init="ortho"):
        super(biggan_deep_generator, self).__init__()
        
        self.dim_z = dim_z
        self.gene_ch = gene_ch
        self.img_size = img_size
        assert self.img_size in [64, 128, 192, 256]
        self.nc = nc #channel
        self.dim_y = dim_y #embedding dimension of y
        self.use_sn = use_sn #use spectral normalization?
        self.use_attn=use_attn #use self-attention?
        self.g_init = g_init #how to initialize the nework
        
        ## defaul setting for initial size and channel multipliers
        if self.img_size == 64:
            self.init_size = 4
            if ch_multi is None:
                ch_multi=[16, 16, 8, 4, 2, 1]
            assert len(ch_multi)>=6
        elif self.img_size == 128:
            self.init_size = 4
            if ch_multi is None:
                ch_multi=[16, 16, 8, 8, 4, 2, 1]
            assert len(ch_multi)>=7    
        elif self.img_size == 192:
            self.init_size = 3
            if ch_multi is None:
                ch_multi=[16, 16, 8, 8, 4, 4, 2, 1]
            assert len(ch_multi)>=8
        elif self.img_size == 256:
            self.init_size = 4
            if ch_multi is None:
                ch_multi=[16, 16, 8, 8, 4, 4, 2, 1]
            assert len(ch_multi)>=8
        self.ch_multi = ch_multi
        
        self.num_blocks = len(ch_multi)-1 #number of residual blocks
        self.affine_input_dim = self.dim_z + self.dim_y
        
        self.linear0 = mylinear(in_features=self.affine_input_dim, out_features=self.gene_ch*self.ch_multi[0]*self.init_size*self.init_size, bias=True, use_sn=self.use_sn)

        self.block10 = GenBlock(gene_ch*ch_multi[0], gene_ch*ch_multi[0], dim_cond=self.affine_input_dim, use_sn=self.use_sn, upsample=False) #4x4 or 3x3
        self.block11 = GenBlock(gene_ch*ch_multi[0], gene_ch*ch_multi[1], dim_cond=self.affine_input_dim, use_sn=self.use_sn, upsample=True) #8x8 or 6x6
        
        self.block20 = GenBlock(gene_ch*ch_multi[1], gene_ch*ch_multi[1], dim_cond=self.affine_input_dim, use_sn=self.use_sn, upsample=False) #8x8 or 6x6
        self.block21 = GenBlock(gene_ch*ch_multi[1], gene_ch*ch_multi[2], dim_cond=self.affine_input_dim, use_sn=self.use_sn, upsample=True) #16x16 or 12x12
        
        self.block30 = GenBlock(gene_ch*ch_multi[2], gene_ch*ch_multi[2], dim_cond=self.affine_input_dim, use_sn=self.use_sn, upsample=False) #16x16 or 12x12
        self.block31 = GenBlock(gene_ch*ch_multi[2], gene_ch*ch_multi[3], dim_cond=self.affine_input_dim, use_sn=self.use_sn, upsample=True) #32x32 or 24x24

        self.block40 = GenBlock(gene_ch*ch_multi[3], gene_ch*ch_multi[3], dim_cond=self.affine_input_dim, use_sn=self.use_sn, upsample=False) #32x32 or 24x24
        self.block41 = GenBlock(gene_ch*ch_multi[3], gene_ch*ch_multi[4], dim_cond=self.affine_input_dim, use_sn=self.use_sn, upsample=True) #64x64 or 48x48
        
        if self.img_size in [64]:
            self.block50 = GenBlock(gene_ch*ch_multi[4], gene_ch*ch_multi[4], dim_cond=self.affine_input_dim, use_sn=self.use_sn, upsample=False) #64x64
            self.block51 = GenBlock(gene_ch*ch_multi[4], gene_ch*ch_multi[-1], dim_cond=self.affine_input_dim, use_sn=self.use_sn, upsample=False) #64x64
        
        if self.img_size in [128]:
            self.block50 = GenBlock(gene_ch*ch_multi[4], gene_ch*ch_multi[4], dim_cond=self.affine_input_dim, use_sn=self.use_sn, upsample=False) #64x64
            self.block51 = GenBlock(gene_ch*ch_multi[4], gene_ch*ch_multi[5], dim_cond=self.affine_input_dim, use_sn=self.use_sn, upsample=True) #128x128
            
            self.block60 = GenBlock(gene_ch*ch_multi[5], gene_ch*ch_multi[5], dim_cond=self.affine_input_dim, use_sn=self.use_sn, upsample=False) #128x128
            self.block61 = GenBlock(gene_ch*ch_multi[5], gene_ch*ch_multi[-1], dim_cond=self.affine_input_dim, use_sn=self.use_sn, upsample=False) #128x128

        if self.img_size in [192, 256]:
            self.block50 = GenBlock(gene_ch*ch_multi[4], gene_ch*ch_multi[4], dim_cond=self.affine_input_dim, use_sn=self.use_sn, upsample=False) #64x64 or 48x48
            self.block51 = GenBlock(gene_ch*ch_multi[4], gene_ch*ch_multi[5], dim_cond=self.affine_input_dim, use_sn=self.use_sn, upsample=True) #128x128 or 96x96
            
            self.block60 = GenBlock(gene_ch*ch_multi[5], gene_ch*ch_multi[5], dim_cond=self.affine_input_dim, use_sn=self.use_sn, upsample=False) #128x128 or 96x96
            self.block61 = GenBlock(gene_ch*ch_multi[5], gene_ch*ch_multi[6], dim_cond=self.affine_input_dim, use_sn=self.use_sn, upsample=True) #256x256 or 192x192
            
            self.block70 = GenBlock(gene_ch*ch_multi[6], gene_ch*ch_multi[6], dim_cond=self.affine_input_dim, use_sn=self.use_sn, upsample=False) #256x256 or 192x192
            self.block71 = GenBlock(gene_ch*ch_multi[6], gene_ch*ch_multi[-1], dim_cond=self.affine_input_dim, use_sn=self.use_sn, upsample=False) #256x256 or 192x192
        
        # self-attention module
        if self.use_attn:
            if self.img_size in [64]:
                self.self_attn = Self_Attn(gene_ch*ch_multi[3], use_sn=self.use_sn) #after block31
            elif self.img_size in [128, 192, 256]:
                self.self_attn = Self_Attn(gene_ch*ch_multi[4], use_sn=self.use_sn) #after block41
        
        self.bn = nn.BatchNorm2d(gene_ch*ch_multi[-1], eps=1e-5, momentum=0.0001, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv2d1 = myconv2d(in_channels=gene_ch*ch_multi[-1], out_channels=self.nc, kernel_size=3, stride=1, padding=1, use_sn=self.use_sn)
        self.tanh = nn.Tanh()
        
        init_weights(self.modules, self.g_init)
        
        
    def forward(self, z, y):
        
        zy = torch.cat([z.view(-1,self.dim_z), y.view(-1,self.dim_y)],dim=1)
        assert zy.size(1)==self.dim_y+self.dim_z
        
        # first linear layer
        out = self.linear0(zy)            # self.init_size x self.init_size
        # reshape
        out = out.view(-1, self.gene_ch*self.ch_multi[0], self.init_size, self.init_size) 
        
        out = self.block10(out, zy)    # 4 x 4 or 3 x 3
        out = self.block11(out, zy)    # 8 x 8 or 6 x 6
        out = self.block20(out, zy)    # 8 x 8 or 6 x 6
        out = self.block21(out, zy)    # 16 x 16 or 12 x 12  
        out = self.block30(out, zy)    # 16 x 16 or 12 x 12
        out = self.block31(out, zy)    # 32 x 32 or 24 x 24
        
        if self.img_size in [64] and self.use_attn:
            out = self.self_attn(out)     # 32 x 32
            
        out = self.block40(out, zy)    # 32 x 32 or 24 x 24
        out = self.block41(out, zy)    # 64 x 64 or 48 x 48
        
        if self.img_size in [128, 192, 256] and self.use_attn:
            out = self.self_attn(out)     # 64 x 64 or 48 x 48
        
        out = self.block50(out, zy) #64x64 or 48 x 48
        out = self.block51(out, zy) #128x128 or 96 x 96
        
        if self.img_size in [128, 192, 256]:
            out = self.block60(out, zy) #64x64 or 48 x 48
            out = self.block61(out, zy) #128x128 or 96 x 96
            
        if self.img_size in [192, 256]:
            out = self.block70(out, zy) #128x128 or 96 x 96
            out = self.block71(out, zy) #256x256 or 192 x 192
        
        out = self.bn(out)
        out = self.relu(out)
        out = self.conv2d1(out)
        out = self.tanh(out)
        return out






######################################################
# discriminator

class DiscOptBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_sn=False, channel_ratio=4):
        super(DiscOptBlock, self).__init__()
        
        self.use_sn = use_sn # use spectral normalization?
        hidden_channels = out_channels // channel_ratio
        assert in_channels != out_channels
        
        self.activation = nn.ReLU(inplace=True)
        
        self.conv2d1 = myconv2d(in_channels=in_channels,
                                out_channels=hidden_channels,
                                kernel_size=1,
                                stride=1,
                                padding=0, 
                                use_sn=use_sn)
        self.conv2d2 = myconv2d(in_channels=hidden_channels,
                                out_channels=hidden_channels,
                                kernel_size=3,
                                stride=1,
                                padding=1, 
                                use_sn=use_sn)
        self.conv2d3 = myconv2d(in_channels=hidden_channels,
                                out_channels=hidden_channels,
                                kernel_size=3,
                                stride=1,
                                padding=1, 
                                use_sn=use_sn)
        self.conv2d4 = myconv2d(in_channels=hidden_channels,
                                out_channels=out_channels,
                                kernel_size=1,
                                stride=1,
                                padding=0, 
                                use_sn=use_sn)
        
        self.conv2d0 = myconv2d(in_channels=in_channels,
                                out_channels=out_channels,
                                kernel_size=1,
                                stride=1,
                                padding=0, 
                                use_sn=use_sn)
        
        self.average_pooling = nn.AvgPool2d(2)
        
    def forward(self, x):
        x0 = x
        x = self.conv2d1(self.activation(x))
        x = self.conv2d2(self.activation(x))
        x = self.conv2d3(self.activation(x))
        x = self.average_pooling(x)
        x = self.conv2d4(self.activation(x))
        
        x0 = self.average_pooling(x0)
        x0 = self.conv2d0(x0)
        
        out = x + x0
        return out
        
        
class DiscBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_sn=False, downsample=True, channel_ratio=4):
        super(DiscBlock, self).__init__()
        
        self.use_sn = use_sn # use spectral normalization?
        self.downsample=downsample 
        hidden_channels = out_channels // channel_ratio
        self.ch_mismatch = True if (in_channels != out_channels) else False
    
        self.activation = nn.ReLU(inplace=True)
        
        self.conv2d1 = myconv2d(in_channels=in_channels,
                                out_channels=hidden_channels,
                                kernel_size=1,
                                stride=1,
                                padding=0, 
                                use_sn=use_sn)
        self.conv2d2 = myconv2d(in_channels=hidden_channels,
                                out_channels=hidden_channels,
                                kernel_size=3,
                                stride=1,
                                padding=1, 
                                use_sn=use_sn)
        self.conv2d3 = myconv2d(in_channels=hidden_channels,
                                out_channels=hidden_channels,
                                kernel_size=3,
                                stride=1,
                                padding=1, 
                                use_sn=use_sn)
        self.conv2d4 = myconv2d(in_channels=hidden_channels,
                                out_channels=out_channels,
                                kernel_size=1,
                                stride=1,
                                padding=0, 
                                use_sn=use_sn)
        
        if self.ch_mismatch or self.downsample:
            self.conv2d0 = myconv2d(in_channels=in_channels,
                                out_channels=out_channels,
                                kernel_size=1,
                                stride=1,
                                padding=0, 
                                use_sn=use_sn)
            
        if self.downsample:
            self.average_pooling = nn.AvgPool2d(2)
            
    def forward(self, x):
        x0 = x
        
        x = self.conv2d1(self.activation(x))
        x = self.conv2d2(self.activation(x))
        x = self.conv2d3(self.activation(x))
        if self.downsample:
            x = self.average_pooling(x)
        x = self.conv2d4(self.activation(x))

        if self.downsample or self.ch_mismatch:
            x0 = self.conv2d0(x0)
            if self.downsample:
                x0 = self.average_pooling(x0)
        out = x + x0
        
        return out


class biggan_deep_discriminator(nn.Module):
    """Discriminator."""

    def __init__(self, dim_y=128, img_size=64, nc=3, disc_ch=32, ch_multi=None, use_sn=True, use_attn=True, d_init="ortho", use_aux_reg=False, use_aux_dre=False, dre_head_arch="MLP3", p_dropout=0.5):
        super(biggan_deep_discriminator, self).__init__()
        
        self.dim_y = dim_y
        self.nc = nc
        self.img_size = img_size
        assert self.img_size in [64, 128, 192, 256]
        self.use_sn = use_sn
        self.use_attn = use_attn
        self.d_init = d_init
        self.use_aux_reg = use_aux_reg
        self.use_aux_dre = use_aux_dre
        self.dre_head_arch = dre_head_arch
        
        ## defaul setting for initial size and channel multipliers
        if self.img_size == 64:
            self.init_size = 4
            if ch_multi is None:
                ch_multi=[1, 2, 4, 8, 16]
            assert len(ch_multi)==5
        elif self.img_size == 128:
            self.init_size = 4
            if ch_multi is None:
                ch_multi=[1, 2, 4, 8, 16, 16]
            assert len(ch_multi)==6    
        elif self.img_size == 192:
            self.init_size = 3
            if ch_multi is None:
                ch_multi=[1, 2, 4, 8, 8, 16, 16]
            assert len(ch_multi)>=7
        elif self.img_size == 256:
            self.init_size = 4
            if ch_multi is None:
                ch_multi=[1, 2, 4, 8, 8, 16, 16]
            assert len(ch_multi)==7
        self.ch_multi = ch_multi

        self.disc_ch = disc_ch
        
        self.input_conv = myconv2d(in_channels=nc, out_channels=disc_ch*ch_multi[0], kernel_size=3, stride=1, padding=1, use_sn=use_sn)
        
        self.opt_block1 = DiscOptBlock(disc_ch*ch_multi[0], disc_ch*ch_multi[1], use_sn=use_sn) #32x32, 64x64, 96x96, 128x128
        self.block1 = DiscBlock(disc_ch*ch_multi[1], disc_ch*ch_multi[1], use_sn=use_sn, downsample=False) #32x32, 64x64, 96x96, 128x128
        
        self.block20 = DiscBlock(disc_ch*ch_multi[1], disc_ch*ch_multi[2], use_sn=use_sn, downsample=True) #16x16, 32x32, 48x48, 64x64
        self.block21 = DiscBlock(disc_ch*ch_multi[2], disc_ch*ch_multi[2], use_sn=use_sn, downsample=False) #16x16, 32x32, 48x48, 64x64
        
        self.block30 = DiscBlock(disc_ch*ch_multi[2], disc_ch*ch_multi[3], use_sn=use_sn, downsample=True) #8x8, 16x16, 24x24, 32x32
        self.block31 = DiscBlock(disc_ch*ch_multi[3], disc_ch*ch_multi[3], use_sn=use_sn, downsample=False) #8x8, 16x16, 24x24, 32x32
        
        if self.img_size in [64]:
            self.block40 = DiscBlock(disc_ch*ch_multi[3], disc_ch*ch_multi[4], use_sn=use_sn, downsample=True) #4x4
            self.block41 = DiscBlock(disc_ch*ch_multi[4], disc_ch*ch_multi[4], use_sn=use_sn, downsample=False)#4x4
        elif self.img_size in [128]:
            self.block40 = DiscBlock(disc_ch*ch_multi[3], disc_ch*ch_multi[4], use_sn=use_sn, downsample=True)#8x8
            self.block41 = DiscBlock(disc_ch*ch_multi[4], disc_ch*ch_multi[4], use_sn=use_sn, downsample=False)#8x8
            self.block50 = DiscBlock(disc_ch*ch_multi[4], disc_ch*ch_multi[5], use_sn=use_sn, downsample=True)#4x4
            self.block51 = DiscBlock(disc_ch*ch_multi[5], disc_ch*ch_multi[5], use_sn=use_sn, downsample=False)#4x4
        elif self.img_size in [192, 256]:
            self.block40 = DiscBlock(disc_ch*ch_multi[3], disc_ch*ch_multi[4], use_sn=use_sn, downsample=True)#16x16, 12x12
            self.block41 = DiscBlock(disc_ch*ch_multi[4], disc_ch*ch_multi[4], use_sn=use_sn, downsample=False)#16x16, 12x12
            self.block50 = DiscBlock(disc_ch*ch_multi[4], disc_ch*ch_multi[5], use_sn=use_sn, downsample=True)#8x8, 6x6
            self.block51 = DiscBlock(disc_ch*ch_multi[5], disc_ch*ch_multi[5], use_sn=use_sn, downsample=False)#8x8, 6x6
            self.block60 = DiscBlock(disc_ch*ch_multi[5], disc_ch*ch_multi[6], use_sn=use_sn, downsample=True)#4x4, 3x3
            self.block61 = DiscBlock(disc_ch*ch_multi[6], disc_ch*ch_multi[6], use_sn=use_sn, downsample=False)#4x4, 3x3
        
        if self.use_attn:
            if self.img_size in [64, 128]:
                self.self_attn = Self_Attn(disc_ch*ch_multi[1], use_sn=self.use_sn) #after block1, at 32x32
            elif self.img_size in [192, 256]:
                self.self_attn = Self_Attn(disc_ch*ch_multi[2], use_sn=self.use_sn) #after block21, at 48x48 or 64x64

        self.relu = nn.ReLU(inplace=True)
        self.linear1 = mylinear(in_features=disc_ch*ch_multi[-1]*self.init_size*self.init_size, out_features=1, use_sn=True)
        self.linear2 = mylinear(self.dim_y, disc_ch*ch_multi[-1]*self.init_size*self.init_size, bias=False, use_sn=True)

        if self.use_aux_reg:
            self.reg_linear = nn.Sequential(
                spectral_norm(nn.Linear(disc_ch*self.ch_multi[-1]*self.init_size*self.init_size, 128)),
                nn.ReLU(),
                spectral_norm(nn.Linear(128, 1)),
                nn.ReLU(),
            )
            
        if use_aux_dre:
            if self.dre_head_arch == "MLP5":
                self.dre_linear = nn.Sequential(
                    nn.Linear(disc_ch*self.ch_multi[-1]*self.init_size*self.init_size+self.dim_y, 2048),
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
                    nn.Linear(disc_ch*self.ch_multi[-1]*self.init_size*self.init_size+self.dim_y, 2048),
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
                    nn.Linear(disc_ch*self.ch_multi[-1]*self.init_size*self.init_size+self.dim_y, 512),
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
                    nn.Linear(disc_ch*self.ch_multi[-1]*self.init_size*self.init_size+self.dim_y, 512),
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
        init_weights(self.modules, self.d_init)
        
    def forward(self, x, y):
        
        h = self.input_conv(x)  
        h = self.opt_block1(h)  
        h = self.block1(h)
        if self.img_size in [64, 128] and self.use_attn:
            h = self.self_attn(h)
        h = self.block20(h)
        h = self.block21(h)
        if self.img_size in [192, 256] and self.use_attn:
            h = self.self_attn(h)    
        h = self.block30(h)
        h = self.block31(h)
        h = self.block40(h)
        h = self.block41(h)
        if self.img_size in [128, 192, 256]:
            h = self.block50(h)
            h = self.block51(h)
        if self.img_size in [256, 192]:
            h = self.block60(h)
            h = self.block61(h)
            
        h = self.relu(h)              # n x disc_ch*ch_multi[-1] x self.init_size x self.init_size
        h = h.view(-1, self.disc_ch*self.ch_multi[-1]*self.init_size*self.init_size)
        
        output1 = torch.squeeze(self.linear1(h)) # n
        # Projection
        h_y = self.linear2(y)   # n x disc_ch*self.ch_multi[-1]
        proj = torch.mul(h, h_y)          # n x disc_ch*self.ch_multi[-1]
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
            feat_cat = torch.cat((h, y), -1)
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
    NC=3
    DIM_Z=128
    DIM_Y=128

    netG = biggan_deep_generator(dim_z=DIM_Z, dim_y=DIM_Y, img_size=IMG_SIZE, nc=NC, gene_ch=64, ch_multi=None, use_sn=True, use_attn=True, g_init="ortho").cuda() # parameters
    netD = biggan_deep_discriminator(dim_y=DIM_Y, img_size=IMG_SIZE, nc=NC, disc_ch=64, ch_multi=None, use_sn=True, use_attn=True, d_init="ortho", use_aux_reg=True, use_aux_dre=True, dre_head_arch="MLP3").cuda() # parameters

    # netG = nn.DataParallel(netG)
    # netD = nn.DataParallel(netD)

    N=4
    z = torch.randn(N, DIM_Z).cuda()
    y = torch.randn(N, DIM_Y).cuda()
    x = netG(z,y)
    print(x.size())
    out = netD(x,y)
    print(out['adv_output'].size())
    print(out['reg_output'].size())
    print(out['dre_output'].size())

    print('G:', get_parameter_number(netG))
    print('D:', get_parameter_number(netD))