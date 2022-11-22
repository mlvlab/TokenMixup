
weightroot="weights"
datadir="/hub_data/imagenet/"

python vit/train.py \
    --eval \
    --run_name imagenet_HVTM \
    --dataset imagenet \
    --model_type ViT-B_16 \
    --pretrained_dir $weightroot/VIT_HVTM.bin \
    --data_dir $datadir \
