import torch
from torch import nn
import torch.nn.functional as F

# from spectral_normalization import SpectralNorm
import numpy as np
from torch.nn.utils import spectral_norm



channels = 1
bias = True

class ConditionalBatchNorm2d(nn.Module):
    def __init__(self, num_features, dim_embed):
        super().__init__()
        self.num_features = num_features
        self.bn = nn.BatchNorm2d(num_features, affine=False)

        # self.embed = nn.Linear(dim_embed, num_features * 2, bias=False)
        # self.embed.weight.data[:, :num_features].normal_(1, 0.02)  # Initialise scale at N(1, 0.02)
        # self.embed.weight.data[:, num_features:].zero_()  # Initialise bias at 0
        # # self.embed = spectral_norm(self.embed) #seems not work

        self.embed_gamma = nn.Linear(dim_embed, num_features, bias=False)
        self.embed_beta = nn.Linear(dim_embed, num_features, bias=False)

    def forward(self, x, y):
        out = self.bn(x)
        # gamma, beta = self.embed(y).chunk(2, 1)
        # out = gamma.view(-1, self.num_features, 1, 1) * out + beta.view(-1, self.num_features, 1, 1)

        gamma = self.embed_gamma(y).view(-1, self.num_features, 1, 1)
        beta = self.embed_beta(y).view(-1, self.num_features, 1, 1)
        out = out + out*gamma + beta

        return out



#########################################################
# genearator
class dcgan_generator(nn.Module):
    def __init__(self, dim_z=128, dim_y=128, nc=3, gene_ch = 64):
        super(dcgan_generator, self).__init__()
        self.dim_z = dim_z
        self.gene_ch = gene_ch
        self.dim_y = dim_y

        self.linear = nn.Linear(dim_z, 4 * 4 * gene_ch * 8) #4*4*512

        self.deconv1 = nn.ConvTranspose2d(gene_ch * 8, gene_ch * 8, kernel_size=4, stride=2, padding=1, bias=bias) #h=2h 8
        self.deconv2 = nn.ConvTranspose2d(gene_ch * 8, gene_ch * 4, kernel_size=4, stride=2, padding=1, bias=bias) #h=2h 16
        self.deconv3 = nn.ConvTranspose2d(gene_ch * 4, gene_ch * 2, kernel_size=4, stride=2, padding=1, bias=bias) #h=2h 32
        self.deconv4 = nn.ConvTranspose2d(gene_ch * 2, gene_ch, kernel_size=4, stride=2, padding=1, bias=bias) #h=2h 64
        self.condbn1 = ConditionalBatchNorm2d(gene_ch * 8, dim_y)
        self.condbn2 = ConditionalBatchNorm2d(gene_ch * 4, dim_y)
        self.condbn3 = ConditionalBatchNorm2d(gene_ch * 2, dim_y)
        self.condbn4 = ConditionalBatchNorm2d(gene_ch, dim_y)
        self.relu = nn.ReLU()

        self.final_conv = nn.Sequential(
            nn.Conv2d(gene_ch, gene_ch, kernel_size=3, stride=1, padding=1, bias=bias), #h=h
            nn.BatchNorm2d(gene_ch),
            nn.ReLU(),
            nn.Conv2d(gene_ch, nc, kernel_size=3, stride=1, padding=1, bias=bias), #h=h
            nn.Tanh()
        )


    def forward(self, z, y):
        z = z.view(-1, self.dim_z)

        out = self.linear(z)
        out = out.view(-1, 8*self.gene_ch, 4, 4)

        out = self.deconv1(out)
        out = self.condbn1(out, y)
        out = self.relu(out)

        out = self.deconv2(out)
        out = self.condbn2(out, y)
        out = self.relu(out)

        out = self.deconv3(out)
        out = self.condbn3(out, y)
        out = self.relu(out)

        out = self.deconv4(out)
        out = self.condbn4(out, y)
        out = self.relu(out)

        out = self.final_conv(out)

        return out

#########################################################
# discriminator
class dcgan_discriminator(nn.Module):
    def __init__(self, dim_y=128, nc=3, disc_ch = 64, use_aux_reg=False, use_aux_dre=False, dre_head_arch="MLP3", p_dropout=0.5):
        super(dcgan_discriminator, self).__init__()
        self.disc_ch = disc_ch
        self.dim_y = dim_y
        self.nc = nc
        
        self.use_aux_reg = use_aux_reg
        self.use_aux_dre = use_aux_dre
        self.dre_head_arch = dre_head_arch

        self.conv = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(self.nc, self.disc_ch, kernel_size=4, stride=2, padding=1, bias=bias), #h=h/2
            nn.BatchNorm2d(self.disc_ch),
            nn.LeakyReLU(0.2, inplace=True),
            # input is disc_ch x 32 x 32
            nn.Conv2d(self.disc_ch, self.disc_ch*2, kernel_size=4, stride=2, padding=1, bias=bias), #h=h/2
            nn.BatchNorm2d(self.disc_ch*2),
            nn.LeakyReLU(0.2, inplace=True),
            # input is (disc_ch*2) x 16 x 16
            nn.Conv2d(self.disc_ch*2, self.disc_ch*4, kernel_size=4, stride=2, padding=1, bias=bias), #h=h/2
            nn.BatchNorm2d(self.disc_ch*4),
            nn.LeakyReLU(0.2, inplace=True),
            # input is (disc_ch*4) x 8 x 8
            nn.Conv2d(self.disc_ch*4, self.disc_ch*8, kernel_size=4, stride=2, padding=1, bias=bias), #h=h/2
            # nn.BatchNorm2d(self.disc_ch*8),
            nn.LeakyReLU(0.2, inplace=True),

            # nn.Conv2d(self.disc_ch*8, self.disc_ch*8, 3, stride=1, padding=1, bias=bias),  #h=h
            # nn.LeakyReLU(0.2, inplace=True)
        )

        self.linear1 = nn.Linear(self.disc_ch*8*4*4, 1, bias=True)
        nn.init.xavier_uniform_(self.linear1.weight.data, 1.)
        self.linear1 = spectral_norm(self.linear1)
        self.linear2 = nn.Linear(self.dim_y, self.disc_ch*8*4*4, bias=False)
        nn.init.xavier_uniform_(self.linear2.weight.data, 1.)
        self.linear2 = spectral_norm(self.linear2)
        
        if use_aux_reg:
            self.reg_linear = nn.Sequential(
                spectral_norm(nn.Linear(self.disc_ch*8*4*4, 128)),
                nn.ReLU(),
                spectral_norm(nn.Linear(128, 1)),
                nn.ReLU(),
            )
        
        if use_aux_dre:
            if self.dre_head_arch == "MLP5":
                self.dre_linear = nn.Sequential(
                    nn.Linear(self.disc_ch*8*4*4+self.dim_y, 2048),
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
                    nn.Linear(self.disc_ch*8*4*4+self.dim_y, 2048),
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
                    nn.Linear(self.disc_ch*8*4*4+self.dim_y, 512),
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
                    nn.Linear(self.disc_ch*8*4*4+self.dim_y, 512),
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


    def forward(self, x, y):

        h = self.conv(x)

        # out = torch.sum(out, dim=(2,3))
        h = h.view(-1, self.disc_ch*8*4*4)
        out_y = torch.sum(h*self.linear2(y), 1, keepdim=True)
        adv_output = self.linear1(h) + out_y
        adv_output = adv_output.view(-1,1)
        
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

    netG = dcgan_generator(dim_z=256, dim_y=128, nc=3, gene_ch = 64).cuda() #131367044 parameters
    netD = dcgan_discriminator(dim_y=128, nc=3, disc_ch = 64, use_aux_reg=True, use_aux_dre=True, dre_head_arch="MLP3").cuda() #162158402 parameters

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