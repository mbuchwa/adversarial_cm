print("\n===================================================================================================")

import argparse
import copy
import gc
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib as mpl
import h5py
import os
import random
from tqdm import tqdm, trange
import torch
import torchvision
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torchvision.utils import save_image
import timeit
from PIL import Image
import sys
from datetime import datetime 

from utils import *
from models import sagan_generator, sagan_discriminator, sngan_generator, sngan_discriminator, biggan_generator, biggan_discriminator, biggan_deep_generator, biggan_deep_discriminator, resnet18_aux_regre
from dataset import LoadDataSet
from label_embedding import LabelEmbed
from trainer import Trainer
from opts import parse_opts
from evaluation.evaluator import Evaluator

##############################################
''' Settings '''
args = parse_opts()

# seeds
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True
cudnn.benchmark = False
np.random.seed(args.seed)


#######################################################################################
'''                                Output folders                                  '''
#######################################################################################
path_to_output = os.path.join(args.root_path, 'output/{}_{}'.format(args.data_name, args.img_size))
os.makedirs(path_to_output, exist_ok=True)

save_setting_folder = os.path.join(path_to_output, "{}".format(args.setting_name))
os.makedirs(save_setting_folder, exist_ok=True)

setting_log_file = os.path.join(save_setting_folder, 'setting_info.txt')
if not os.path.isfile(setting_log_file):
    logging_file = open(setting_log_file, "w")
    logging_file.close()
with open(setting_log_file, 'a') as logging_file:
    logging_file.write("\n===================================================================================================")
    print(args, file=logging_file)

save_results_folder = os.path.join(save_setting_folder, 'results')
os.makedirs(save_results_folder, exist_ok=True)

path_to_fake_data = os.path.join(save_results_folder, 'fake_data')
os.makedirs(path_to_fake_data, exist_ok=True)


#######################################################################################
'''                                Make dataset                                     '''
#######################################################################################

dataset = LoadDataSet(data_name=args.data_name, data_path=args.data_path, min_label=args.min_label, max_label=args.max_label, img_size=args.img_size, max_num_img_per_label=args.max_num_img_per_label, num_img_per_label_after_replica=args.num_img_per_label_after_replica, imbalance_type=args.imb_type)
    
train_images, train_labels, train_labels_norm = dataset.load_train_data()
num_classes = dataset.num_classes

_, _, eval_labels = dataset.load_evaluation_data()

#######################################################################################
'''                           Compute Vicinal Params                                '''
#######################################################################################

unique_labels_norm = np.sort(np.array(list(set(train_labels_norm))))

if args.kernel_sigma<0:
    std_label = np.std(train_labels_norm)
    args.kernel_sigma = 1.06*std_label*(len(train_labels_norm))**(-1/5)

    print("\n Use rule-of-thumb formula to compute kernel_sigma >>>")
    print("\r The std of {} labels is {:.4f} so the kernel sigma is {:.4f}".format(len(train_labels_norm), std_label, args.kernel_sigma))
##end if

if args.kappa<0:
    n_unique = len(unique_labels_norm)

    diff_list = []
    for i in range(1,n_unique):
        diff_list.append(unique_labels_norm[i] - unique_labels_norm[i-1])
    kappa_base = np.abs(args.kappa)*np.max(np.array(diff_list))

    if args.threshold_type=="hard":
        args.kappa = kappa_base
    else:
        args.kappa = 1/kappa_base**2
##end if

print("\r Kappa:{:.4f}".format(args.kappa))

vicinal_params = {
    "kernel_sigma": args.kernel_sigma,
    "kappa": args.kappa,
    "threshold_type": args.threshold_type,
    "nonzero_soft_weight_threshold": args.nonzero_soft_weight_threshold,
    "use_ada_vic":args.use_ada_vic,
    "ada_vic_type":args.ada_vic_type,
    "min_n_per_vic": args.min_n_per_vic,
    "ada_eps":1e-5,
    "use_symm_vic": args.use_symm_vic,
}


#######################################################################################
'''                             label embedding method                              '''
#######################################################################################

dataset_embed = LoadDataSet(data_name=args.data_name, data_path=args.data_path, min_label=args.min_label, max_label=args.max_label, img_size=args.img_size, max_num_img_per_label=args.max_num_img_per_label, num_img_per_label_after_replica=0, imbalance_type=args.imb_type)

label_embedding = LabelEmbed(dataset=dataset_embed, path_y2h=path_to_output+'/model_y2h', path_y2cov=path_to_output+'/model_y2cov', y2h_type="resnet", y2cov_type="sinusoidal", h_dim = args.dim_y, cov_dim = args.img_size**2*args.num_channels, nc=args.num_channels)
fn_y2h = label_embedding.fn_y2h



#######################################################################################
'''                                 Model Config                                    '''
#######################################################################################

if args.ch_multi_g is not None:
    ch_multi_g = (args.ch_multi_g).split("_")
    ch_multi_g = [int(dim) for dim in ch_multi_g]
if args.ch_multi_d is not None:
    ch_multi_d = (args.ch_multi_d).split("_")
    ch_multi_d = [int(dim) for dim in ch_multi_d]

if args.net_name.lower() == "sngan":
    netG = sngan_generator(dim_z=args.dim_z, dim_y=args.dim_y, nc=args.num_channels, img_size=args.img_size, gene_ch=args.gene_ch, ch_multi=args.ch_multi_g)
    netD = sngan_discriminator(dim_y=args.dim_y, nc=args.num_channels, img_size=args.img_size, disc_ch=args.disc_ch, ch_multi=args.ch_multi_d, use_aux_reg=args.use_aux_reg_branch, use_aux_dre = args.use_dre_reg, dre_head_arch=args.dre_head_arch)
elif args.net_name.lower() == "sagan":
    netG = sagan_generator(dim_z=args.dim_z, dim_y=args.dim_y, nc=args.num_channels, img_size=args.img_size, gene_ch=args.gene_ch, ch_multi=args.ch_multi_g)
    netD = sagan_discriminator(dim_y=args.dim_y, nc=args.num_channels, img_size=args.img_size, disc_ch=args.disc_ch, ch_multi=args.ch_multi_d, use_aux_reg=args.use_aux_reg_branch, use_aux_dre = args.use_dre_reg, dre_head_arch=args.dre_head_arch)
elif args.net_name.lower() == "biggan":
    netG = biggan_generator(dim_z=args.dim_z, dim_y=args.dim_y, img_size=args.img_size, nc=args.num_channels, gene_ch=args.gene_ch, ch_multi=args.ch_multi_g, use_sn=args.use_sn, use_attn=args.use_attn, g_init="ortho")
    netD = biggan_discriminator(dim_y=args.dim_y, img_size=args.img_size, nc=args.num_channels, disc_ch=args.disc_ch, ch_multi=args.ch_multi_d, use_sn=args.use_sn, use_attn=args.use_attn, d_init="ortho", use_aux_reg=args.use_aux_reg_branch, use_aux_dre = args.use_dre_reg, dre_head_arch=args.dre_head_arch)
elif args.net_name.lower() == "biggan-deep":
    netG = biggan_deep_generator(dim_z=args.dim_z, dim_y=args.dim_y, img_size=args.img_size, nc=args.num_channels, gene_ch=args.gene_ch, ch_multi=args.ch_multi_g, use_sn=args.use_sn, use_attn=args.use_attn, g_init="ortho")
    netD = biggan_deep_discriminator(dim_y=args.dim_y, img_size=args.img_size, nc=args.num_channels, disc_ch=args.disc_ch, ch_multi=args.ch_multi_d, use_sn=args.use_sn, use_attn=args.use_attn, d_init="ortho", use_aux_reg=args.use_aux_reg_branch, use_aux_dre = args.use_dre_reg, dre_head_arch=args.dre_head_arch)
elif args.net_name.lower() == "dcgan":
    netG = sngan_generator(dim_z=args.dim_z, dim_y=args.dim_y, nc=args.num_channels, img_size=args.img_size, gene_ch=args.gene_ch)
    netD = sngan_discriminator(dim_y=args.dim_y, nc=args.num_channels, img_size=args.img_size, disc_ch=args.disc_ch, use_aux_reg=args.use_aux_reg_branch, use_aux_dre = args.use_dre_reg, dre_head_arch=args.dre_head_arch)
else:
    raise ValueError("Not Supported Network!")

print('\r netG size:', get_parameter_number(netG))
print('\r netD size:', get_parameter_number(netD))


## independent auxiliary regressor
if args.use_aux_reg_model:
    aux_reg_net = resnet18_aux_regre(nc=args.num_channels)
    path_to_ckpt = os.path.join(path_to_output, "aux_reg_model")
    if args.data_name in ["RC-49_imb"]:
        path_to_ckpt += "/{}".format(args.imb_type)
    path_to_ckpt += "/ckpt_resnet18_epoch_200.pth"
    checkpoint = torch.load(path_to_ckpt, weights_only=True)
    aux_reg_net.load_state_dict(checkpoint['net_state_dict'])
    aux_reg_net.eval()   
else:
    aux_reg_net = None

#######################################################################################
'''                                  Training                                      '''
#######################################################################################

aux_loss_params = {
    #===========================
    # auxiliary regression loss for both D and G
    "use_aux_reg_branch":args.use_aux_reg_branch,
    "use_aux_reg_model":args.use_aux_reg_model,
    "aux_reg_loss_type": args.aux_reg_loss_type,
    "aux_reg_loss_ei_hinge_factor":args.aux_reg_loss_ei_hinge_factor,
    "aux_reg_loss_huber_delta":args.aux_reg_loss_huber_delta,
    "aux_reg_loss_huber_quantile":args.aux_reg_loss_huber_quantile,
    "weight_d_aux_reg_loss": args.weight_d_aux_reg_loss,
    "weight_g_aux_reg_loss": args.weight_g_aux_reg_loss,
    "aux_reg_net": aux_reg_net,
    #===========================
    # density density ratio model training and auxiliary penalty for G
    "use_dre_reg": args.use_dre_reg,
    "dre_lambda": args.dre_lambda,
    "weight_d_aux_dre_loss": args.weight_d_aux_dre_loss,
    "weight_g_aux_dre_loss": args.weight_g_aux_dre_loss,
    "do_dre_ft": args.do_dre_ft, #finetuning dre branch after the GAN training
    "dre_ft_niters": args.dre_ft_niters,
    "dre_ft_lr": args.dre_ft_lr,
    "dre_ft_batch_size": args.dre_ft_batch_size,
}

trainer = Trainer(
    data_name=args.data_name,
    train_images=train_images,
    train_labels=train_labels_norm,
    eval_labels = dataset.fn_normalize_labels(eval_labels),
    net_name=args.net_name,
    netG=netG,
    netD=netD,
    fn_y2h=fn_y2h,
    vicinal_params=vicinal_params,
    aux_loss_params = aux_loss_params,
    img_size=args.img_size,
    img_ch=args.num_channels,
    results_folder=save_results_folder,
    dim_z = args.dim_z,
    niters = args.niters,
    resume_iter = args.resume_iter,
    num_D_steps = args.num_D_steps, 
    batch_size_disc = args.batch_size_disc,
    batch_size_gene = args.batch_size_gene,
    lr_g = args.lr_g,
    lr_d = args.lr_d,
    loss_type = args.loss_type,
    save_freq = args.save_freq,
    sample_freq = args.sample_freq,
    num_grad_acc_d = args.num_grad_acc_d,
    num_grad_acc_g = args.num_grad_acc_g,
    max_grad_norm = args.max_grad_norm,
    nrow_visual = 10,
    use_amp = args.use_amp,
    mixed_precision_type = args.mixed_precision_type,
    adam_betas = (0.5, 0.999),
    use_ema = args.use_ema,
    ema_update_after_step = args.ema_update_after_step,
    ema_update_every = args.ema_update_every,
    ema_decay = args.ema_decay,
    use_diffaug = args.use_diffaug,
    diffaug_policy = args.diffaug_policy,
    exp_seed = args.seed,
    num_workers = None,
)

start = timeit.default_timer()
print("\n")
print("Begin Training:")
trainer.train()
stop = timeit.default_timer()
print("End training; Time elapses: {}s. \n".format(stop - start))

if args.do_dre_ft:
    trainer.finetune_cdre()




#######################################################################################
'''                         Sampling and evaluation                                 '''
#######################################################################################


print("\n Start sampling fake images from the model >>>")

## initialize evaluator
evaluator = Evaluator(dataset=dataset, trainer=trainer, args=args, device=trainer.device) #, root_path=args.root_path

## initialize evaluation models, prepare for evaluation
if args.data_name in ["RC-49","RC-49_imb"]:
    eval_data_name = "RC49"
else:
    eval_data_name = args.data_name
conduct_import_codes = "from evaluation.eval_models.{}.metrics_{}x{} import ResNet34_class_eval, ResNet34_regre_eval, encoder".format(eval_data_name, args.img_size, args.img_size)
print("\r"+conduct_import_codes)
exec(conduct_import_codes)
# for FID
PreNetFID = encoder(dim_bottleneck=512)
PreNetFID = nn.DataParallel(PreNetFID)
# for Diversity
if args.data_name in ["UTKFace", "RC-49", "RC-49_imb", "SteeringAngle"]:
    PreNetDiversity = ResNet34_class_eval(num_classes=num_classes, ngpu = torch.cuda.device_count())
else:
    PreNetDiversity = None
# for LS
PreNetLS = ResNet34_regre_eval(ngpu = torch.cuda.device_count())


## dump fake data in h5 files
if args.dump_fake_for_h5:
    path_to_h5files = os.path.join(path_to_fake_data, 'h5')
    os.makedirs(path_to_h5files, exist_ok=True)
    evaluator.dump_h5_files(output_path=path_to_h5files)

## dump for niqe computation
if args.dump_fake_for_niqe:
    if args.niqe_dump_path=="None":
        dump_fake_images_folder = os.path.join(path_to_fake_data, 'png')
    else:
        dump_fake_images_folder = args.niqe_dump_path + '/fake_images'
    os.makedirs(dump_fake_images_folder, exist_ok=True)
    evaluator.dump_png_images(output_path=dump_fake_images_folder)


## start computing evaluation metrics
if args.do_eval:
    now = datetime.now()
    time_str = now.strftime("%Y-%m-%d_%H-%M-%S")
    eval_results_path = os.path.join(save_setting_folder, "eval_{}".format(time_str))
    os.makedirs(eval_results_path, exist_ok=True)
    evaluator.compute_metrics(eval_results_path, PreNetFID, PreNetDiversity, PreNetLS)




print("\n===================================================================================================")