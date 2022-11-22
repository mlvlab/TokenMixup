# TokenMixup

This is the official implementation of the NeurIPS 2022 paper, "[TokenMixup: Efficient Attention-guided Data Augmentation for Transformers](https://arxiv.org/abs/2210.07562)" by H. Choi, J. Choi, and H. J. Kim.

<!-- ⠀            |  ⠀
:-------------------------:|:-------------------------:
![htm_gif](https://raw.githubusercontent.com/mlvlab/TokenMixup/main/assets/HTM.gif)  |  ![vtm_gif](https://raw.githubusercontent.com/mlvlab/TokenMixup/main/assets/VTM.gif)  -->
<p float="middle">
  <img src="https://raw.githubusercontent.com/mlvlab/TokenMixup/main/assets/HTM.gif" width="49%" />
  <img src="https://raw.githubusercontent.com/mlvlab/TokenMixup/main/assets/VTM.gif" width="49%" /> 
</p>

## Setup
* Clone repository

```
git clone https://github.com/mlvlab/TokenMixup.git
cd TokenMixup
```

Below are setup details for experiments in our paper. If you want to jump to using TokenMixup for your model, have a look at [this section](#usage).

* Setup conda environment
```
conda env create --file env.yaml
conda activate tokenmixup
```

* Install packages that require manual setup
```
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip install tensorboard
pip install setuptools==59.5.0

cd experiments
# Current (Nov 2022) version of apex is not compatible with our torch version.
# For now, skip direct installation and use the apex file provided in this repo.
# git clone https://github.com/NVIDIA/apex
# cd apex
cd apex_copy
python setup.py install
cd ../..
```

* Setup datasets

CIFAR10 and CIFAR100 does not require dataset setup. The code will do it for you once you set a root directory. If you want to play around with ViT, download [Imagenet-1K](https://image-net.org/) and save it in an accessible directory.


* Prepare weights

If you want to train the ViT model, download the official pretrained ViT weight for initialization.

```
mkdir experiments/weights
cd experiments/weights
wget https://storage.googleapis.com/vit_models/imagenet21k+imagenet2012/ViT-B_16-224.npz
cd ../..
```

To evaluate our trained models, download the weight you wish to evaluate from the [Model Zoo](#model-zoo), and place it in the `experiments/weights/` folder.


## Run Experiments

The shell files are in the `experiments/run/` directory. 
In each file, update the **weightroot** and **datadir** values.

The following is an example with CCT_7-3x1 + HTM on CIFAR100.

```
cd experiments

# Inference
sh run/eval_cct_7-3x1_cifar100_HTM.sh

# Train
sh run/train_cct_7-3x1_cifar100_HTM.sh
```

All other scripts follow the same file name style. 
Note, experiments were trained with several different GPU settings.


## Usage

First, copy our `tokenmixup/` folder to an accessible directory. Follow along the following to get started!

### (Horizontal) TokenMixup
1. Import module and wrap the encoder layer you wish to apply (Horizontal) TokenMixup. Note that `HorizontalTokenMixupLayer` can be applied to multiple layers. Also, please refer to the `tokenmixup/horizontal.py` for input parameter explanations.
```
from tokenmixup import HorizontalTokenMixupLayer

encoder_layer1 = TransformerEncoderLayer( ... )
encoder_layer2 = HorizontalTokenMixupLayer(
                      TransformerEncoderLayer( ... ),
                      rho =  ... ,
                      tau = ... ,
                      use_scorenet = ... ,
                      ...
                )
```

2. Have your model take the data and target as the first two inputs, and output predictions and the mixed target.
The target should be one-hot, but you don't need to do anything in the model with the target variable. 
Just make sure it flows through the encoder layers and is finally output safe and sound. The rest will be taken care of by our module!
```
pred_logit, target = your_model(X, target, ... )
```

3. If you are using ScoreNet (by setting `use_scorenet=True`), the outputted "target" will be a tuple of (mixed target, scorenet loss).
Add the loss to your model loss as follows. Note, the lambda value for scorenet is applied within the module.
There is no need for additional lambda terms.
```
target, scorenet_loss = target
loss = your_loss_function(pred_logit, target) + scorenet_loss.mean()
```

4. Inside the encoder layer class, define a function named `get_attention_map`. 
The function should receive the exact same input as the forward function, and output only the attention map.
```
class TransformerEncoderLayer(nn.Module):
    def __init__(self, ... ):
        self.self_attn = Attention( ... )
        ...

    def forward(self,  X, target, *args, **kwargs):
        ...

    def get_attention_map(self, X, target, *args, **kwargs) :
        # something like...
        X = (whatever you may need before self attention)
        X, attn = self.self_attn(X)
        return attn
```

4. That's pretty much it! When you run our module with `verbose=True` (default), you will be informed with 
min (avg) sample difficulty and max (avg) saliency difference values, along with mixed sample and token counts. 
These should help you set appropriate values for tau (difficulty threshold) and rho (saliency difference threshold).


### Vertical TokenMixup
1. Import module and wrap the self attention module where you wish to apply Vertical TokenMixup, and provide which layer to apply
VTM, current layer index, etc.
Note, you need to do this in all encoder layers, to make it work properly. Also, please refer to `tokenmixup/vertical.py` for input parameter explanations.
```
from tokenmixup import VerticalTokenMixupLayer
class TransformerEncoderLayer(nn.Module):
    def __init__(self, ... ):
        # originally,  self.self_attn = SelfAttention( ... )
        self.self_attn = VerticalTokenMixupLayer(
                            SelfAttention( ... )
                            apply_layers = [ ... ],
                            layer_index = ... ,
                            kappa = ... ,
                            vtm_attn_dim = ... ,
                            vtm_attn_numheads = ...
                        )
        ...

    def forward(self,  ... ):
        ...
```

2. call the reset function after finishing iteration of encoder layers. You only need to call it once, with any encoder layer. For instance,
```
for layer in encoder_layers :
    X = layer(X)
layer.self_attn.reset()
```

**DISCLAIMER**
> The way VTM uses normalization slightly differs from the experiment codes used for the paper experiments. Originally, normalization was applied to the concatenated token set of input X and multi-scale tokens. But now, prenorm needs to be respectively applied prior to VerticalTokenMixupLayer.

3. That's all. When you run our module with `verbose=True` (default), you will be informed with the number of multi-scale tokens that are concatenated to the key & value tokens.


## Model Zoo

| Baseline Model | TokenMixup Type | Dataset Type | Top 1 Accuracy  |  Checkpoint   |
|:--------------:|:---------------:|:------------:|:---------------:|:------------:|
|  CCT_7-3x1  |    HTM    |   CIFAR10   |  97.57   | [drive](https://drive.google.com/file/d/1Q9yQ2-oof39fVHTKQtffmn1p-8pcgaDI/view?usp=sharing)  |
|  CCT_7-3x1  |    VTM    |   CIFAR10   |  97.78  | [drive](https://drive.google.com/file/d/12MAgOFGYX9RmyA7TOkMvL2a01zX7NieM/view?usp=sharing)  |
|  CCT_7-3x1  | HTM + VTM |   CIFAR10   |  97.75  | [drive](https://drive.google.com/file/d/1nh1ayrG3aIhCowncN82Pt0kQE3CoXYpn/view?usp=sharing)  |
|  CCT_7-3x1  |    HTM    |   CIFAR100  |  83.56  | [drive](https://drive.google.com/file/d/1XRUtJGNSD68Mo4HqoRGoBLu8zaEITnGY/view?usp=sharing)  |
|  CCT_7-3x1  |    VTM    |   CIFAR100  |  83.54  | [drive](https://drive.google.com/file/d/1HjDDwfgmGxlFZHiSjaMQ5y4e3Y571ceb/view?usp=sharing)  |
|  CCT_7-3x1  | HTM + VTM |   CIFAR100  |  83.57  | [drive](https://drive.google.com/file/d/1TdjMa1Gt5I2OQ-GbbhbhvW7TK7r4V8ey/view?usp=sharing)  |
|ViT_B/16-224 |    HTM    | Imagenet1K  |  82.37  | [drive](https://drive.google.com/file/d/1-GFm5siNYVDmWKXcqeb3ci6jMNL2CIiA/view?usp=sharing)  |
|ViT_B/16-224 |    VTM    | Imagenet1K  |  82.30  | [drive](https://drive.google.com/file/d/1iCU8ofa5Uoig1rP5VCkowWSUAKAPIWLw/view?usp=sharing)  |
|ViT_B/16-224 | HTM + VTM | Imagenet1K  |  82.32  | [drive](https://drive.google.com/file/d/1rThWY--Zw73Fv8i8f6X-BHlpi6-0X1HT/view?usp=sharing)  |


We have noticed that our weights' inference performance varies slightly by GPU.
Note that we evaluated our weights identically on a single RTX 2080ti GPU for fair comparisons.


## Contributors
- [Hyeong Kyu Choi](https://github.com/imhgchoi)
- [Joonmyung Choi](https://github.com/pizard)


## UPDATES
**2022.11.22** Initial code release


## Citation

```
@inproceedings{choi2022tokenmixup,
  title={TokenMixup: Efficient Attention-guided Token-level Data Augmentation for Transformers},
  author={Choi, Hyeong Kyu and Choi, Joonmyung and Kim, Hyunwoo J.},
  booktitle={Advances in Neural Information Processing Systems},
  year={2022}
}
```

## License
Code is released under [MIT License](https://github.com/mlvlab/TokenMixup/blob/main/LICENSE).

> Copyright (c) 2022-present Korea University Research and Business Foundation & MLV Lab  
