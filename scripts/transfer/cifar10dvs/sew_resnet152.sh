python transfer.py\
    --batch_size 60 \
    --dataset 'cifar10_dvs' \
    --root '/home/haohq/datasets/CIFAR10DVS' \
    --nsteps 8 \
    --nclasses 10 \
    --batch_size 128 \
    --model 'sew_resnet152' \
    --pretrained_path '/home/haohq/SNN-Trans-Assess-main/output/pretrain/sew_resnet152_es_imagenet_b6_lr0.01_T8_CE_adam_mom0.9_step3_gamma0.3_cnfADD/checkpoint/checkpoint_epoch10_valacc26.86.pth' \
    --device_id 4 \
    --output_dir 'output/cifar10dvs/'
    