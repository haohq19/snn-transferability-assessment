python transfer.py\
    --batch_size 60 \
    --dataset 'cifar10_dvs' \
    --root '/home/haohq/datasets/CIFAR10DVS' \
    --nsteps 8 \
    --nclasses 10 \
    --batch_size 128 \
    --model 'sew_resnet50' \
    --pretrained_path '/home/haohq/SNN-Trans-Assess-main/weights/sew50_checkpoint_319.pth'\
    --device_id 2 \
    --output_dir 'output/cifar10dvs/'
    