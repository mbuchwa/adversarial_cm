from .resnet_y2h import ResNet34_embed_y2h, model_y2h
from .resnet_y2cov import ResNet34_embed_y2cov, model_y2cov
from .sngan import sngan_generator, sngan_discriminator
from .sagan import sagan_generator, sagan_discriminator
from .biggan import biggan_generator, biggan_discriminator
from .biggan_deep import biggan_deep_generator, biggan_deep_discriminator
from .dcgan import dcgan_generator, dcgan_discriminator
from .resnet_aux_regre import resnet18_aux_regre, resnet34_aux_regre, resnet50_aux_regre