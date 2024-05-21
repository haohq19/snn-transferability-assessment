torchrun --nproc_per_node=8 pretrain.py\
    --dataset 'es_imagenet' \
    --root 'path to the dataset' \
    --nsteps 8 \
    --num_classes 1000 \
    --batch_size 60 \
    --model 'spiking_resnet18' \
    --connect_f 'ADD' \
    --nepochs 10 \
    --nworkers 32 \
    --lr 0.01 \
    --output_dir '' \
    --save_freq 1 \
    --step_size 3 \
    --gamma 0.3 \
    --sync_bn \
    --backend 'nccl'