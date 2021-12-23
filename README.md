# Barbershorp
Reproducing code of hair style replacement method from [Barbershorp](https://arxiv.org/abs/2106.01505).
Also reproduces [II2S](http://arxiv.org/abs/2012.09036), an improved version of Image2StyleGAN.

## Requirements
Tested on Windows, which includes:
```Shell
numpy              1.17
opencv-python      4.5.1
pytorchcv          0.0.67
torch              1.7.1+cu110
tqdm               4.6
```
Opencv is only for visualization, not necessary for computation.

## Usage
Get StyleGAN model file `stylegan2-ffhq-config-f.pt` as described in https://github.com/rosinality/stylegan2-pytorch. 

Run II2S to get latent code(s) of input image(s). 
```Shell
python II2S.py --ckpt stylegan2-ffhq-config-f.pt --size 1024 --w_plus 'ImagePath1.png' 'ImagePath2.png'
```
Specify the related paths and parameters in `config.py`, then run `barbershorp.py`:
```Shell
python barbershorp.py
```

## Code References
stylegan2_pytorch code was taken from https://github.com/rosinality/stylegan2-pytorch

