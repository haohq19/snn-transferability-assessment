python -m torch.distributed.run --nproc_per_node=8 train.py\
    --batch_size 40 \
    --dataset 'es_imagenet' \
    --root '/home/haohq/datasets/ESImageNet' \
    --nsteps 8 \
    --nclasses 1000 \
    --model 'spiking_resnet34' \
    --nepochs 10 \
    --optim 'Adam'\
    --lr 0.01 \
    --sched 'StepLR' \
    --step_size 3 \
    --gamma 0.3 \
    --save_freq 1 \
    --sync_bn