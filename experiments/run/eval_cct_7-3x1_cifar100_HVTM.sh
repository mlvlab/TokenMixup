
datadir="../data"

python cct/eval.py \
    --config cct/configs/pretrained/cct_cifar100_HVTM.yaml \
    --initial-checkpoint weights/cct_cifar100_hvtm.pth.tar \
    $datadir
