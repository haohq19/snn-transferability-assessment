python transfer.py\
    --batch_size 60 \
    --dataset 'cifar10_dvs' \
    --root '/home/haohq/datasets/CIFAR10DVS' \
    --nsteps 8 \
    --nclasses 10 \
    --batch_size 128 \
    --model 'spiking_resnet18' \
    --pretrained_path '/home/haohq/SNN-Trans-Assess-main/output/pretrain/spiking_resnet18_es_imagenet_b60_lr0.01_T8_CE_adam_mom0.9_step3_gamma0.3_cnfADD/checkpoint/checkpoint_epoch10_valacc31.56.pth' \
    --device_id 5 \
    --output_dir 'output/cifar10dvs/'
    