python transfer.py\
    --batch_size 60 \
    --dataset 'n_caltech101' \
    --root '/home/haohq/datasets/NCaltech101' \
    --nsteps 8 \
    --nclasses 101 \
    --batch_size 128 \
    --model 'sew_resnet50' \
    --pretrained_path '/home/haohq/SNN-Trans-Assess-main/weights/sew50_checkpoint_319.pth'\
    --device_id 2 \
    --output_dir 'output/ncaltech101/'
    