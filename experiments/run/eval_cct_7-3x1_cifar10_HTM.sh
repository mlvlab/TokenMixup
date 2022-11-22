
datadir="../data"

python cct/eval.py \
    --config cct/configs/pretrained/cct_cifar10_HTM.yaml \
    --initial-checkpoint weights/cct_cifar10_htm.pth.tar \
    $datadir
