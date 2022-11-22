
datadir="../data"

python cct/eval.py \
    --config cct/configs/pretrained/cct_cifar10_VTM.yaml \
    --initial-checkpoint weights/cct_cifar10_vtm.pth.tar \
    $datadir
