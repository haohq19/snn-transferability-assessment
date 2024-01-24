python transfer.py\
    --batch_size 60 \
    --dataset 'cifar10_dvs' \
    --root '/home/haohq/datasets/CIFAR10DVS' \
    --nsteps 8 \
    --nclasses 10 \
    --batch_size 128 \
    --model 'sew_resnet152' \
    --pretrained_path '/home/haohq/SNN-Trans-Assess-main/weights/sew152_checkpoint_319.pth'\
    --device_id 4 \
    --output_dir 'output/cifar10dvs/'
    