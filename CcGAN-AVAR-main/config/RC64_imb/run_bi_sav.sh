#!/bin/bash

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export CUDA_VISIBLE_DEVICES=0

ROOT_PATH="<YOUR PATH>"
DATA_PATH="<YOUR PATH>"
NIQE_PATH="<YOUR PATH>"

DATA_NAME="RC-49_imb"
SETTING="bi_sav"

SEED=2024
MIN_LABEL=0
MAX_LABEL=90.0
IMG_SIZE=64
IMB_TYPE="dualmodal"

BATCH_SIZE_G=256
BATCH_SIZE_D=256
NUM_D_STEPS=2
SIGMA=-1
KAPPA=-2
LR_G=1e-4
LR_D=1e-4
NUM_ACC_D=1
NUM_ACC_G=1

NET_NAME="SNGAN"
LOSS_TYPE="hinge"
THRESH_TYPE="soft"

DIM_GAN=256
DIM_Y=128

NITERS=50000
RESUME_ITER=0
python main.py \
    --setting_name "$SETTING" --data_name "$DATA_NAME" --imb_type $IMB_TYPE \
    --root_path "$ROOT_PATH" --data_path "$DATA_PATH" --seed "$SEED" \
    --min_label "$MIN_LABEL" --max_label "$MAX_LABEL" --img_size "$IMG_SIZE" \
    --net_name "$NET_NAME" --dim_z "$DIM_GAN" --dim_y "$DIM_Y" \
    --gene_ch 64 --disc_ch 48 \
    --niters "$NITERS" --resume_iter $RESUME_ITER --loss_type "$LOSS_TYPE" --num_D_steps "$NUM_D_STEPS" \
    --save_freq 10000 --sample_freq 5000 \
    --batch_size_disc "$BATCH_SIZE_D" --batch_size_gene "$BATCH_SIZE_G" \
    --lr_g "$LR_G" --lr_d "$LR_D" \
    --num_grad_acc_d "$NUM_ACC_D" --num_grad_acc_g "$NUM_ACC_G" \
    --kernel_sigma "$SIGMA" --threshold_type $THRESH_TYPE --kappa "$KAPPA" \
    --use_diffaug --diffaug_policy color,translation,cutout \
    --use_ema --use_amp --max_grad_norm 1.0 \
    --use_ada_vic --ada_vic_type vanilla --min_n_per_vic 30 --use_symm_vic \
    --use_aux_reg_branch --use_aux_reg_model \
    --aux_reg_loss_type ei_hinge --weight_d_aux_reg_loss 1.0 --weight_g_aux_reg_loss 1.0 \
    --use_dre_reg --dre_lambda 1e-2 --weight_d_aux_dre_loss 0.5 --weight_g_aux_dre_loss 0.5 \
    --do_eval \
    --samp_batch_size 200 --eval_batch_size 200 \
    2>&1 | tee output_${DATA_NAME}_${SETTING}.txt

    # --do_eval \
    # --dump_fake_for_niqe --niqe_dump_path $NIQE_PATH \
    # --use_ema