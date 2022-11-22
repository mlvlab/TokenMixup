
datadir="../data"

python cct/eval.py \
    --config cct/configs/pretrained/cct_cifar100_VTM.yaml \
    --initial-checkpoint weights/cct_cifar100_vtm.pth.tar \
    $datadir
