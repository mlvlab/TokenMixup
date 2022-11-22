
weightroot="weights"
datadir="/hub_data/imagenet/"


python vit/train.py \
    --run_name imagenet_HTM \
    --epochs 30 \
    --train_batch_size 504 \
    --horizontal-mixup True \
    --horizontal-layer 3 \
    --rho 0.01 \
    --tau 0.1 \
    --scorenet-stopgrad True \
    --scorenet-lambda 1.0 \
    --scorenet-train True \
    --vertical-mixup False \
    --vertical-stopgrad True \
    --learning_rate 0.015 \
    --min_lr 0.01 \
    --dataset imagenet \
    --model_type ViT-B_16 \
    --pretrained_dir $weightroot/ViT-B_16-224.npz \
    --data_dir $datadir
