python transfer.py\
    --batch_size 60 \
    --dataset 'n_mnist' \
    --root '/home/haohq/datasets/NMNIST' \
    --nsteps 8 \
    --nclasses 10 \
    --batch_size 128 \
    --model 'spiking_resnet50' \
    --pretrained_path  '/home/haohq/SNN-Trans-Assess-main/output/pretrain/spiking_resnet50_es_imagenet_b12_lr0.01_T8_CE_adam_mom0.9_step3_gamma0.3_cnfADD/checkpoint/checkpoint_epoch10_valacc16.93.pth' \
    --device_id 7 \
    --output_dir 'output/nmnist/'
    