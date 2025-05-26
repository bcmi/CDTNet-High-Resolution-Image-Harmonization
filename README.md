# CDTNet-High-Resolution-Image-Harmonization


This is the official repository for the following paper:

> **High-Resolution Image Harmonization via Collaborative Dual Transformations**  [[arXiv]](https://arxiv.org/abs/2109.06671)<br>
>
> Wenyan Cong, Xinhao Tao, Li Niu, Jing Liang, Xuesong Gao, Qihao Sun, Liqing Zhang<br>
> Accepted by **CVPR2022**.

**Our CDTNet(sim) has been integrated into our image composition toolbox libcom https://github.com/bcmi/libcom. Welcome to visit and try ＼(^▽^)／** 

**This is the first paper focusing on high-resolution image harmonization. We divide image harmonization methods into pixel-to-pixel transformation and color-to-color transformation.** We propose CDTNet to combine these two coherently in an end-to-end framework. As shown in the image below, our CDTNet consists of a low-resolution generator for pixel-to-pixel transformation, a color mapping module for RGB-to-RGB transformation, and a refinement module to take advantage of both. **For efficiency, you can use CDTNet(sim) which only has color-to-color transformation.**

<img src='examples/network.jpg' align="center" width=700>

Note that CDTNet(sim) only supports global color transformation. To achieve local (spatially-variant) color transformation, you can refer to more recent works like [PCTNet](https://libcom.readthedocs.io/en/latest/image_harmonization.html).

## Getting Started

### Prerequisites
Please refer to [iSSAM](https://github.com/saic-vul/image_harmonization) and [3D LUT](https://github.com/HuiZeng/Image-Adaptive-3DLUT) for guidance on setting up the environment.

### Installation
+ Clone this repo: 
```
git clone https://github.com/bcmi/CDTNet-High-Resolution-Image-Harmonization
cd ./CDTNet-High-Resolution-Image-Harmonization
```
+ Download the [iHarmony4](https://github.com/bcmi/Image-Harmonization-Dataset-iHarmony4) dataset, and configure the paths to the datasets in [config.yml](./config.yml).

### Training
If you want to train CDTNet-512 on 2048*2048 HAdobe5K training set with 4 LUTs and pre-trained pixel-to-pixel transformation model "issam256.pth", you can run this command:
```
python3 train.py models/CDTNet.py --gpus=0 --workers=10 --exp_name=CDTNet_1024 --datasets HAdobe5k --batch_size=4 --hr_w 1024 --hr_h 1024 --lr 256 --weights ./issam256.pth --n_lut 4
```

We have also provided some commands in the "train.sh" for your convenience.

### Testing
If you want to test CDTNet-512 on 2048*2048 HAdobe5K test set with the "HAdobe5k_2048.pth" checkpoint and save the results in "CDTNet_2048_result", you can run this command:
```
python3 evaluate_model.py CDTNet ./HAdobe5k_2048.pth --gpu 0 --datasets HAdobe5k --hr_w 2048 --hr_h 2048 --lr 512 --save_dir ./CDTNet_2048_result
```

We have also provided some commands in the "test.sh" for your convenience.

### Prediction
If you want to make predictions using your own dataset which the composite images are in ./predict_images and the masks are in ./masks using CDTNet-512 on 2048*2048 with the "HAdobe5k_2048.pth" checkpoint and save the results in "CDTNet_2048_generate", you can run this command:
```
python3 scripts/predict_for_dir.py CDTNet ./HAdobe5k_2048.pth --images ./predict_images --masks ./predict_masks --gpu 0 --hr_h 2048 --hr_w 2048 --lr 512 --results-path ./CDTNet_2048_generate
```

We have also provided some commands in the "predict.sh" for your convenience.

## Datasets

### 1. HAdobe5k

HAdobe5k is one of the four synthesized sub-datasets in [iHarmony4](https://github.com/bcmi/Image-Harmonization-Dataset-iHarmony4) dataset, which is the benchmark dataset for image harmonization. Specifically, HAdobe5k is generated based on [MIT-Adobe FiveK](<http://data.csail.mit.edu/graphics/fivek/>) dataset and contains 21597 image triplets (composite image, real image, mask) as shown below, where 19437 triplets are used for training and 2160 triplets are used for test. Official training/test split could be found in **[Baidu Cloud](https://pan.baidu.com/s/1qVqsSf0gFjBlfP8xNiNGpQ?pwd=84rb)** or [**Dropbox**](https://www.dropbox.com/scl/fi/lcjrnxfle43daf2l0jgcq/HAdobe5k.zip?rlkey=vekbegch1uh945hsg92dc35h5&st=ezxi90j2&dl=0).

MIT-Adobe FiveK provides with 6 retouched versions for each image, so we manually segment the foreground region and exchange foregrounds between 2 versions to generate composite images. High-resolution images in HAdobe5k sub-dataset are with random resolution from 1296 to 6048, which could be downloaded from **[Baidu Cloud](https://pan.baidu.com/s/1qVqsSf0gFjBlfP8xNiNGpQ?pwd=84rb)** or [**Dropbox**](https://www.dropbox.com/scl/fi/lcjrnxfle43daf2l0jgcq/HAdobe5k.zip?rlkey=vekbegch1uh945hsg92dc35h5&st=ezxi90j2&dl=0).

<img src='examples/hadobe5k.png' align="center" width=600>

### 2. 100 High-Resolution Real Composite Images

Considering that the composite images in HAdobe5k are synthetic composite images, we additionally provide 100 high-resolution real composite images for qualitative comparison in real scenarios with image pairs (composite image, mask), which are generated based on [Open Image Dataset V6](https://storage.googleapis.com/openimages/web/index.html) dataset and [Flickr](https://www.flickr.com). 

Open Image Dataset V6 contains ~9M images with 28M instance segmentation annotations of 350 categories, where enormous images are collected from Flickr and with high resolution. So the foreground images are collected from the whole Open Image Dataset V6, where the provided instance segmentations are used to crop the foregrounds. The background images are collected from both Open Image Dataset V6 and Flickr, considering the resolutions and semantics. Then cropped foregrounds and background images are combined using PhotoShop, leading to obviously inharmonious composite images.

100 high-resolution real composite images are with random resolution from 1024 to 6016, which could be downloaded from **[Baidu Cloud](https://pan.baidu.com/s/1fTfLBMxb7sAKtbpQVsfh8g)** (access code: vnrp) or [**Dropbox**](https://www.dropbox.com/scl/fo/h7scj5yq6x22rxwqnao40/AFJixrNw9mD742h8qj-vG-0?rlkey=0tov1jwskt96p9fqxadgaqecj&st=0w6n4lcd&dl=0).

<img src='examples/hr_real_comp_100.jpg' align="center" width=700>



## Results

#### 1. High-resolution (1024&times;1024 and 2048&times;2048) results on HAdobe5k test set
We test our CDTNet on 1024&times;1024 and 2048&times;2048 images from HAdobe5k dataset and report the harmonization performance based on MSE, PSNR, fMSE,  and SSIM. Here we also release all harmonized results on both resolutions. Due to JPEG compression, the performance tested on our provided results would be not surprisingly worse than our reported performance.

<table class="tg">
  <tr>
    <th class="tg-0pky" align="center">Image Size</th>
    <th class="tg-0pky" align="center">Model</th>
    <th class="tg-0pky" align="center">MSE</th>
    <th class="tg-0pky" align="center">PSNR</th>
    <th class="tg-0pky" align="center">fMSE</th>
    <th class="tg-0pky" align="center">SSIM</th>
    <th class="tg-0pky" align="center">Test Images Download</th>
  </tr>
  <tr>
    <td class="tg-0pky" align="center">1024&times;1024</td>
    <td class="tg-0pky" align="center">CDTNet-256</td>
    <td class="tg-0pky" align="center">21.24</td>
    <td class="tg-0pky" align="center">38.77</td>
    <td class="tg-0pky" align="center">152.13</td>
    <td class="tg-0pky" align="center">0.9868</td>
    <td class="tg-0pky" align="center"><a href="https://pan.baidu.com/s/1ShSJb-0V0SELB6QEXPHsPQ">Baidu Cloud</a> (access code: i8l1) or <a href="https://www.dropbox.com/scl/fo/42gzfuv7xtcu6fbamvng3/AGSy_U_kWSCE_TR8BB63pLw?rlkey=gwj664fr2sqo6uok9gdf7hqvy&st=gjkknp0v&dl=0">Dropbox</a> </td>
  </tr>
  <tr>
    <td class="tg-0pky" align="center">2048&times;2048</td>
    <td class="tg-0pky" align="center">CDTNet-512</td>
    <td class="tg-0pky" align="center">20.82</td>
    <td class="tg-0pky" align="center">38.34</td>
    <td class="tg-0pky" align="center">155.24</td>
    <td class="tg-0pky" align="center">0.9847</td>
      <td class="tg-0pky" align="center"><a href="https://pan.baidu.com/s/1qJ12D0U04UDDPAe8lLRpwg">Baidu Cloud</a> (access code: rj9p) or <a href="https://www.dropbox.com/scl/fi/0510ew72g0z02wv4obgtd/CDTNet_2048_result.zip?rlkey=mblnpp4thn75pc6lcitwcjzuu&st=toeqjn6l&dl=0">Dropbox</a>   </td>
  </tr>
</table>

We show several results on 1024&times;1024 resolution below, where yellow boxes zoom in the particular regions for a better observation.

<img src='examples/examples_1024.png' align="center" width=700>

#### 2. High-resolution (1024&times;1024) results on 100 real composite images 

We test our CDTNet on 100 high-resolution real composite images as mentioned above, and provide the results on [Baidu Cloud](https://pan.baidu.com/s/1RX9ltfA0HskI06THbhpHmA) (access code: lr7k) or [Dropbox](https://www.dropbox.com/scl/fo/x7ofeyn8981s7553q6dtn/AKcMXcTde0BvytVuTN2D9rM?rlkey=vheumcxp4m5ifphmgxczbs67w&st=yqi0eicv&dl=0).

#### 3. Low-resolution (256&times;256) results on iHarmony4 test set

We also test our CDTNet on 256&times;256 images from iHarmony4 dataset. We also provide all harmonized results on [Baidu Cloud](https://pan.baidu.com/s/1CLGeV9BhqSjAbLY_FivhMw) (access code: l7gh) or [Dropbox](https://www.dropbox.com/scl/fi/28ubfhxn2sz0rp978ohmi/CDTNet_256_result.zip?rlkey=kwfsh9yz4la2u6yfdh66cjje0&st=eyj79hwp&dl=0).

#### 4. Low-resolution (256&times;256) results on 99 real composite images

We also test our CDTNet on another 99 real composite images used in [previous works](https://github.com/bcmi/Image-Harmonization-Dataset-iHarmony4), and provide the results on [Baidu Cloud](https://pan.baidu.com/s/1djl8dHqzbZg893fUf6zImw) (access code: i6e8) or [Dropbox](https://www.dropbox.com/scl/fi/adv1t5yoz9c8xux3rq9ti/99-composite-result.zip?rlkey=6nm69z7n908zikca2xj28xfo8&st=m2xxpcll&dl=0).


## Other Resources

+ [Awesome-Image-Harmonization](https://github.com/bcmi/Awesome-Image-Harmonization)
+ [Awesome-Image-Composition](https://github.com/bcmi/Awesome-Object-Insertion)


## Acknowledgement<a name="codesource"></a> 

Our code is heavily borrowed from [iSSAM](https://github.com/saic-vul/image_harmonization) and [3D LUT](https://github.com/HuiZeng/Image-Adaptive-3DLUT).
