python transfer.py\
    --batch_size 60 \
    --dataset 'n_caltech101' \
    --root '/home/haohq/datasets/NCaltech101' \
    --nsteps 8 \
    --nclasses 101 \
    --batch_size 256 \
    --model 'spiking_resnet18' \
    --pretrained_path '/home/haohq/SNN-Trans-Assess-main/weights/spiking_resnet_18_checkpoint_319.pth'\
    --device_id 5 \
    --output_dir 'output/ncaltech101/'
    