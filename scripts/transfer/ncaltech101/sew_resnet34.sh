python transfer.py\
    --batch_size 60 \
    --dataset 'n_caltech101' \
    --root '/home/haohq/datasets/NCaltech101' \
    --nsteps 8 \
    --nclasses 101 \
    --batch_size 256 \
    --model 'sew_resnet34' \
    --pretrained_path '/home/haohq/SNN-Trans-Assess-main/output/pretrain/sew_resnet34_es_imagenet_b40_lr0.01_T8_CE_adam_mom0.9_step3_gamma0.3_cnfADD/checkpoint/checkpoint_epoch10_valacc35.07.pth' \
    --device_id 1 \
    --output_dir 'output/ncaltech101/'
    