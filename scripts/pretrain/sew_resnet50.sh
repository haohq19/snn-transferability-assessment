python -m torch.distributed.run --nproc_per_node=8 pretrain.py\
    --dataset 'es_imagenet' \
    --root '/home/haohq/datasets/ESImageNet-old' \
    --nsteps 8 \
    --nclasses 1000 \
    --batch_size 12 \
    --model 'sew_resnet50' \
    --connect_f 'ADD' \
    --nepochs 100 \
    --nworkers 16 \
    --lr 0.01 \
    --optim 'Adam'\
    --output_dir 'outputs/pretrain/' \
    --save_freq 10 \
    --sched 'StepLR' \
    --step_size 25 \
    --gamma 0.3 \
    --sync_bn