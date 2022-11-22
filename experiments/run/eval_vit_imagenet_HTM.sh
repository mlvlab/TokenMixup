
weightroot="weights"
datadir="/hub_data/imagenet/"

python vit/train.py \
    --eval \
    --run_name imagenet_HTM \
    --dataset imagenet \
    --model_type ViT-B_16 \
    --pretrained_dir $weightroot/VIT_HTM.bin \
    --data_dir $datadir \
