python transfer.py\
    --batch_size 60 \
    --dataset 'n_caltech101' \
    --root '/home/haohq/datasets/NCaltech101' \
    --nsteps 8 \
    --nclasses 101 \
    --batch_size 256 \
    --model 'sew_resnet34' \
    --pretrained_path '/home/haohq/SNN-Trans-Assess-main/weights/sew34_checkpoint_319.pth'\
    --device_id 1 \
    --output_dir 'output/ncaltech101/'
    