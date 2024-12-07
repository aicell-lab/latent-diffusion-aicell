## Overview

This is a repo for latent diffusion models from 2021. We are especially interested in the autoencoder parts for now.

The base for this is https://github.com/CompVis/latent-diffusion, but we also install https://github.com/CompVis/taming-transformers as a module. These two together contain code for training VAEs and VQVAEs, but the problem is that the code is old and uses old versions of all packages.

You can see the README below "insert README".

You can see the folder structure here:

```
➜  latent-diffusion-aicell git:(main) ✗ tree -L 3
.
├── LICENSE
├── README.md
├── assets
│   ├── a-painting-of-a-fire.png
│   ├── a-photograph-of-a-fire.png
│   ├── a-shirt-with-a-fire-printed-on-it.png
│   ├── a-shirt-with-the-inscription-'fire'.png
│   ├── a-watercolor-painting-of-a-fire.png
│   ├── birdhouse.png
│   ├── fire.png
│   ├── inpainting.png
│   ├── modelfigure.png
│   ├── rdm-preview.jpg
│   ├── reconstruction1.png
│   ├── reconstruction2.png
│   ├── results.gif
│   ├── the-earth-is-on-fire,-oil-on-canvas.png
│   ├── txt2img-convsample.png
│   └── txt2img-preview.png
├── configs
│   ├── autoencoder
│   │   ├── autoencoder_kl_16x16x16.yaml
│   │   ├── autoencoder_kl_32x32x4.yaml
│   │   ├── autoencoder_kl_64x64x3.yaml
│   │   └── autoencoder_kl_8x8x64.yaml
│   ├── latent-diffusion
│   │   ├── celebahq-ldm-vq-4.yaml
│   │   ├── cin-ldm-vq-f8.yaml
│   │   ├── cin256-v2.yaml
│   │   ├── ffhq-ldm-vq-4.yaml
│   │   ├── lsun_bedrooms-ldm-vq-4.yaml
│   │   ├── lsun_churches-ldm-kl-8.yaml
│   │   └── txt2img-1p4B-eval.yaml
│   └── retrieval-augmented-diffusion
│       └── 768x768.yaml
├── data
│   ├── DejaVuSans.ttf
│   ├── example_conditioning
│   │   ├── superresolution
│   │   └── text_conditional
│   ├── imagenet_clsidx_to_label.txt
│   ├── imagenet_train_hr_indices.p
│   ├── imagenet_val_hr_indices.p
│   ├── index_synset.yaml
│   └── inpainting_examples
│       ├── 6458524847_2f4c361183_k.png
│       ├── 6458524847_2f4c361183_k_mask.png
│       ├── 8399166846_f6fb4e4b8e_k.png
│       ├── 8399166846_f6fb4e4b8e_k_mask.png
│       ├── alex-iby-G_Pk4D9rMLs.png
│       ├── alex-iby-G_Pk4D9rMLs_mask.png
│       ├── bench2.png
│       ├── bench2_mask.png
│       ├── bertrand-gabioud-CpuFzIsHYJ0.png
│       ├── bertrand-gabioud-CpuFzIsHYJ0_mask.png
│       ├── billow926-12-Wc-Zgx6Y.png
│       ├── billow926-12-Wc-Zgx6Y_mask.png
│       ├── overture-creations-5sI6fQgYIuo.png
│       ├── overture-creations-5sI6fQgYIuo_mask.png
│       ├── photo-1583445095369-9c651e7e5d34.png
│       └── photo-1583445095369-9c651e7e5d34_mask.png
├── environment.yaml
├── ldm
│   ├── data
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── imagenet.py
│   │   └── lsun.py
│   ├── lr_scheduler.py
│   ├── models
│   │   ├── autoencoder.py
│   │   └── diffusion
│   ├── modules
│   │   ├── attention.py
│   │   ├── diffusionmodules
│   │   ├── distributions
│   │   ├── ema.py
│   │   ├── encoders
│   │   ├── image_degradation
│   │   ├── losses
│   │   └── x_transformer.py
│   └── util.py
├── main.py
├── models
│   ├── first_stage_models
│   │   ├── kl-f16
│   │   ├── kl-f32
│   │   ├── kl-f4
│   │   ├── kl-f8
│   │   ├── vq-f16
│   │   ├── vq-f4
│   │   ├── vq-f4-noattn
│   │   ├── vq-f8
│   │   └── vq-f8-n256
│   └── ldm
│       ├── bsr_sr
│       ├── celeba256
│       ├── cin256
│       ├── ffhq256
│       ├── inpainting_big
│       ├── layout2img-openimages256
│       ├── lsun_beds256
│       ├── lsun_churches256
│       ├── semantic_synthesis256
│       ├── semantic_synthesis512
│       └── text2img256
├── notebook_helpers.py
├── notes.md
├── scripts
│   ├── download_first_stages.sh
│   ├── download_models.sh
│   ├── inpaint.py
│   ├── knn2img.py
│   ├── latent_imagenet_diffusion.ipynb
│   ├── sample_diffusion.py
│   ├── train_searcher.py
│   └── txt2img.py
└── setup.py

45 directories, 74 files
```

## training commands

Here's the provided script for training the autoencoder:
`CUDA_VISIBLE_DEVICES=<GPU_ID> python scripts/sample_diffusion.py -r models/ldm/<model_spec>/model.ckpt -l <logdir> -n <\#samples> --batch_size <batch_size> -c <\#ddim steps> -e <\#eta>`

And to just run on CPU and verify that non-GPU stuff is working, we'd run:
`python main.py --base configs/autoencoder/autoencoder_kl_8x8x64_mnist.yaml -t --devices cpu`

This does NOT work right now! There are so many package versions issues that it's probably better to take a step back and rewrite what's needed with modern package versions, which leads us to the refactor.

## Refactor

The first goal is to get `python main.py --base configs/autoencoder/autoencoder_kl_8x8x64.yaml -t --devices cpu` working. We will iteratively modify the code until we get this working.

First step is to look at environment.yaml, and update everything to modern versions. This will results in errors in the code, which we need to fix.

Files that are definitely important are:

```
- main.py
- ldm/models/autoencoder.py
- the config files at config/autoencoder
- the data files in ldm/data
```

### Important

We go through the code step by step.

### cluster stuff

# rsync is preferred as it's more reliable and shows progress

rsync -av --progress \
 --exclude 'logs/' \
 --exclude '\*.pyc' \
 --exclude '**pycache**' \
 --exclude '.git' \
 . x_aleho@berzelius1.nsc.liu.se:/proj/aicell/users/x_aleho/latent-diffusion-aicell/

# Set conda to create environments in project space

export CONDA_ENVS_PATH=/proj/aicell/users/x_aleho/conda_envs
mkdir -p /proj/aicell/users/x_aleho/conda_envs

# Now create environment as before

module load Mambaforge/23.3.1-1-hpc1-bdist
mamba env create -f environment.yaml

rsync -av x_aleho@berzelius1.nsc.liu.se:/proj/aicell/users/x_aleho/latent-diffusion-aicell/logs /Users/lapuerta/aicell/latent-diffusion-aicell

rsync -av x_aleho@berzelius1.nsc.liu.se:/proj/aicell/users/x_aleho/latent-diffusion-aicell/logs/2024-12-07T10-15-57_autoencoder_kl_8x8x64_mnist_gpu /Users/lapuerta/aicell/latent-diffusion-aicell/logs/
