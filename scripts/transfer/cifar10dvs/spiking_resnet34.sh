python transfer.py\
    --batch_size 60 \
    --dataset 'cifar10_dvs' \
    --root '/home/haohq/datasets/CIFAR10DVS' \
    --nsteps 8 \
    --nclasses 10 \
    --batch_size 256 \
    --model 'spiking_resnet34' \
    --pretrained_path '/home/haohq/SNN-Trans-Assess-main/output/pretrain/spiking_resnet34_es_imagenet_b40_lr0.01_T8_CE_adam_mom0.9_step3_gamma0.3_cnfADD/checkpoint/checkpoint_epoch10_valacc23.76.pth' \
    --device_id 6 \
    --output_dir 'output/cifar10dvs/'
    