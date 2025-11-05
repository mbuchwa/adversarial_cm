import argparse

def parse_opts():
    parser = argparse.ArgumentParser()

    ''' Overall Settings '''
    parser.add_argument('--setting_name', type=str, default='Setup1')
    parser.add_argument('--root_path', type=str, default='')
    parser.add_argument('--data_path', type=str, default='')
    parser.add_argument('--seed', type=int, default=2025, metavar='S', help='random seed (default: 2025)')
    parser.add_argument('--num_workers', type=int, default=0)

    ''' Dataset '''
    parser.add_argument('--data_name', type=str, default='RC-49_imb', choices=["RC-49", "UTKFace", "Cell200", "SteeringAngle","RC-49_imb", "Cell200_imb"])
    parser.add_argument('--imb_type', type=str, default='unimodal', choices=['unimodal', 'dualmodal', 'trimodal', 'standard', 'none']) #none means using all data
    parser.add_argument('--min_label', type=float, default=0.0)
    parser.add_argument('--max_label', type=float, default=90.0)
    parser.add_argument('--num_channels', type=int, default=3, metavar='N')
    parser.add_argument('--img_size', type=int, default=64)
    parser.add_argument('--max_num_img_per_label', type=int, default=2**20, metavar='N')
    parser.add_argument('--num_img_per_label_after_replica', type=int, default=0, metavar='N')

    ''' GAN settings '''
    # model config
    parser.add_argument('--net_name', type=str, default='SNGAN')
    # parser.add_argument('--net_name', type=str, default='SNGAN', choices=['SAGAN','SNGAN', 'SNGAN_v2','BigGAN', 'BigGAN-deep', 'DCGAN'])
    parser.add_argument('--dim_z', type=int, default=256, help='Latent dimension of GAN')
    parser.add_argument('--gene_ch', type=int, default=64)
    parser.add_argument('--disc_ch', type=int, default=64)
    parser.add_argument('--ch_multi_g', type=str, default=None)
    parser.add_argument('--ch_multi_d', type=str, default=None)
    parser.add_argument('--use_sn', action='store_true', default=False) #use spectral normalization, effective for biggan
    parser.add_argument('--use_attn', action='store_true', default=False) #use self-attention, effective for biggan
    
    # label embedding config
    parser.add_argument('--embed_type', type=str, default='resnet', choices=['resnet', 'sinusoidal', 'gaussian']) 
    parser.add_argument('--dim_y', type=int, default=128) #dimension of the embedding space

    # training config
    parser.add_argument('--niters', type=int, default=10000, help='number of iterations')
    parser.add_argument('--resume_iter', type=int, default=0)
    parser.add_argument('--lr_g', type=float, default=1e-4, help='learning rate for generator')
    parser.add_argument('--lr_d', type=float, default=1e-4, help='learning rate for discriminator')
    parser.add_argument('--batch_size_disc', type=int, default=64)
    parser.add_argument('--batch_size_gene', type=int, default=64)
    parser.add_argument('--num_D_steps', type=int, default=1, help='number of Ds updates in one iteration')
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    
    parser.add_argument('--save_freq', type=int, default=2000, help='frequency of saving checkpoints')
    parser.add_argument('--sample_freq', type=int, default=2000, help='frequency of visualization')

    parser.add_argument('--use_amp', action='store_true', default=False) #use mixed precision
    parser.add_argument('--mixed_precision_type', type=str, default='fp16', choices=['no', 'fp16', 'bf16'])

    # gradient accumulation
    parser.add_argument('--num_grad_acc_d', type=int, default=1)
    parser.add_argument('--num_grad_acc_g', type=int, default=1)

    # vicinal loss
    parser.add_argument('--loss_type', type=str, default='hinge')
    parser.add_argument('--kernel_sigma', type=float, default=-1.0,
                        help='If kernel_sigma<0, then use rule-of-thumb formula to compute the sigma.')
    parser.add_argument('--threshold_type', type=str, default='hard', choices=['soft', 'hard'])
    parser.add_argument('--kappa', type=float, default=-1)
    parser.add_argument('--nonzero_soft_weight_threshold', type=float, default=1e-3,
                        help='threshold for determining nonzero weights for SVDL; we neglect images with too small weights')
    
    # DiffAugment setting
    parser.add_argument('--use_diffaug', action='store_true', default=False) #use DiffAugment
    parser.add_argument('--diffaug_policy', type=str, default='color,translation,cutout') #DiffAugment policy
    
    # Exponential Moving Average
    parser.add_argument('--use_ema', action='store_true', default=False)
    parser.add_argument('--ema_update_after_step', type=int, default=0)
    parser.add_argument('--ema_update_every', type=int, default=10)
    parser.add_argument('--ema_decay', type=float, default=0.9999)
    
    # Adaptive vicinity
    parser.add_argument('--use_ada_vic', action='store_true', default=False) #use adaptive vicinity
    parser.add_argument('--ada_vic_type', type=str, default='vanilla', choices=["vanilla", "hybrid"]) #adaptive vicinity type
    parser.add_argument('--min_n_per_vic', type=int, default=50) #minimum sample size for each vicinity
    parser.add_argument('--use_symm_vic', action='store_true', default=False) #use symmetric adaptive vicinity
    
    # Auxiliary regression loss
    parser.add_argument('--use_aux_reg_branch', action='store_true', default=False) #whether discriminator has an extra branch for regression
    parser.add_argument('--use_aux_reg_model', action='store_true', default=False) #whether we need a pre-trained independent model for regression
    parser.add_argument('--aux_reg_loss_type', type=str, default='huber', choices=['mse', 'ei_hinge', 'huber']) #mse, vicinal mse, epsilon-insensitive hinge loss
    parser.add_argument('--aux_reg_loss_ei_hinge_factor', type=float, default=1.0)
    parser.add_argument('--aux_reg_loss_huber_delta', type=float, default=-1) #if negative, then use adaptive huber loss
    parser.add_argument('--aux_reg_loss_huber_quantile', type=float, default=0.9)
    parser.add_argument('--weight_d_aux_reg_loss', type=float, default=0.0)
    parser.add_argument('--weight_g_aux_reg_loss', type=float, default=0.0)
            
    # Auxiliary dre loss
    parser.add_argument('--use_dre_reg', action='store_true', default=False)
    parser.add_argument('--dre_lambda', type=float, default=1e-2, help='penalty param for DRE')
    parser.add_argument('--weight_d_aux_dre_loss', type=float, default=0.0)
    parser.add_argument('--weight_g_aux_dre_loss', type=float, default=0.0)
    parser.add_argument('--dre_head_arch', type=str, default='MLP3')
    
    parser.add_argument('--do_dre_ft', action='store_true', default=False) #finetune the DRE branch of the discriminator
    parser.add_argument('--dre_ft_niters', type=int, default=1000, help='number of iterations for finetuning DRE branch')
    parser.add_argument('--dre_ft_lr', type=float, default=1e-4, help='learning rate for finetuning DRE branch')
    parser.add_argument('--dre_ft_batch_size', type=int, default=128, metavar='N')
        
    
    ''' Sampling and Evaluation ''' 
    # Some configurations are implemented in evaluator.py
    parser.add_argument('--samp_batch_size', type=int, default=200)
    parser.add_argument('--do_subsampling', action='store_true', default=False)
    parser.add_argument('--do_eval', action='store_true', default=False)
    parser.add_argument('--dump_fake_for_h5', action='store_true', default=False)
    parser.add_argument('--dump_fake_for_niqe', action='store_true', default=False)
    parser.add_argument('--niqe_dump_path', type=str, default='None') 
    parser.add_argument('--eval_batch_size', type=int, default=200)

    args = parser.parse_args()

    return args


