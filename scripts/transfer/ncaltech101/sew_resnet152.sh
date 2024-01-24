python transfer.py\
    --batch_size 60 \
    --dataset 'n_caltech101' \
    --root '/home/haohq/datasets/NCaltech101' \
    --nsteps 8 \
    --nclasses 101 \
    --batch_size 128 \
    --model 'sew_resnet152' \
    --pretrained_path '/home/haohq/SNN-Trans-Assess-main/weights/sew152_checkpoint_319.pth'\
    --device_id 4 \
    --output_dir 'output/ncaltech101/'
    