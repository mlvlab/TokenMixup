
datadir="../data"

python cct/eval.py \
    --config cct/configs/pretrained/cct_cifar10_HVTM.yaml \
    --initial-checkpoint weights/cct_cifar10_hvtm.pth.tar \
    $datadir
