
datadir="../data"

python cct/eval.py \
    --config cct/configs/pretrained/cct_cifar100_HTM.yaml \
    --initial-checkpoint weights/cct_cifar100_htm.pth.tar \
    $datadir
