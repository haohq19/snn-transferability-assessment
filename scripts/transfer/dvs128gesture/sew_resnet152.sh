python transfer.py\
    --batch_size 60 \
    --dataset 'dvs128_gesture' \
    --root '/home/haohq/datasets/DVS128Gesture' \
    --nsteps 8 \
    --nclasses 11 \
    --batch_size 128 \
    --model 'sew_resnet152' \
    --pretrained_path '/home/haohq/SNN-Trans-Assess-main/output/pretrain/sew_resnet152_es_imagenet_b6_lr0.01_T8_CE_adam_mom0.9_step3_gamma0.3_cnfADD/checkpoint/checkpoint_epoch10_valacc26.86.pth' \
    --device_id 4 \
    --output_dir 'output/dvs128gesture/'
    