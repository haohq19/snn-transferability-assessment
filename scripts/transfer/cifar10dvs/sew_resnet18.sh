python transfer.py\
    --batch_size 60 \
    --dataset 'cifar10_dvs' \
    --root '/home/haohq/datasets/CIFAR10DVS' \
    --nsteps 8 \
    --nclasses 10 \
    --batch_size 256 \
    --model 'sew_resnet18' \
    --pretrained_path '/home/haohq/SNN-Trans-Assess-main/weights/sew18_checkpoint_319.pth'\
    --device_id 0 \
    --output_dir 'output/cifar10dvs/'
    