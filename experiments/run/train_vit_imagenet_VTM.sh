
weightroot="weights"
datadir="/hub_data/imagenet/"


python -m torch.distributed.launch --nproc_per_node=4 \
    vit/train.py \
    --run_name imagenet_VTM \
    --epochs 30 \
    --train_batch_size 126 \
    --horizontal-mixup False \
    --scorenet-stopgrad True \
    --scorenet-lambda 0.0 \
    --scorenet-train False \
    --vertical-mixup True \
    --vertical-layer 3 \
    --kappa 3 \
    --vertical-stopgrad False \
    --learning_rate 0.015 \
    --min_lr 0.01 \
    --dataset imagenet \
    --model_type ViT-B_16 \
    --pretrained_dir $weightroot/ViT-B_16-224.npz \
    --data_dir $datadir
