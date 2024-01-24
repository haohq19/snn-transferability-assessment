python transfer.py\
    --batch_size 60 \
    --dataset 'dvs128_gesture' \
    --root '/home/haohq/datasets/DVS128Gesture' \
    --nsteps 8 \
    --nclasses 11 \
    --batch_size 128 \
    --model 'sew_resnet101' \
    --pretrained_path '/home/haohq/SNN-Trans-Assess-main/weights/sew101_checkpoint_319.pth'\
    --device_id 3 \
    --output_dir 'output/dvs128gesture/'
    