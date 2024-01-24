python transfer.py\
    --batch_size 60 \
    --dataset 'n_mnist' \
    --root '/home/haohq/datasets/NMNIST' \
    --nsteps 8 \
    --nclasses 10 \
    --batch_size 256 \
    --model 'spiking_resnet34' \
    --pretrained_path '/home/haohq/SNN-Trans-Assess-main/weights/spiking_resnet_34_checkpoint_319.pth'\
    --device_id 6 \
    --output_dir 'output/nmnist/'
    