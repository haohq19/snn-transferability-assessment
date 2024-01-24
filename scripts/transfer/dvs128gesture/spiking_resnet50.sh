python transfer.py\
    --batch_size 60 \
    --dataset 'dvs128_gesture' \
    --root '/home/haohq/datasets/DVS128Gesture' \
    --nsteps 8 \
    --nclasses 11 \
    --batch_size 128 \
    --model 'spiking_resnet50' \
    --pretrained_path '/home/haohq/SNN-Trans-Assess-main/weights/spiking_resnet_50_checkpoint_319.pth'\
    --device_id 7 \
    --output_dir 'output/dvs128gesture/'
    