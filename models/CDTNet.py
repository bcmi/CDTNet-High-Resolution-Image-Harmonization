from functools import partial

import torch
from torchvision import transforms
from easydict import EasyDict as edict
from albumentations import HorizontalFlip, Resize, RandomResizedCrop

from iharm.data.compose import ComposeDataset
from iharm.data.hdataset import HDataset
from iharm.data.transforms import HCompose
from iharm.engine.simple_trainer import SimpleHTrainer
from iharm.model import initializer
from iharm.model.base import CDTNet
from iharm.model.losses import MaskWeight_MSE, MSE
from iharm.model.metrics import DenormalizedMSEMetric, DenormalizedPSNRMetric, PSNRMetric, MSEMetric
from iharm.utils.log import logger
from iharm.model.modeling.lut import weights_init_normal_classifier

def train(cfg):
    model, model_cfg = init_model(cfg)
    model_train(model, cfg, model_cfg)


def init_model(cfg):
    model_cfg = edict()
    model_cfg.crop_size = (cfg.hr, cfg.hr)
    model_cfg.input_normalization = {
        'mean': [.485, .456, .406],
        'std': [.229, .224, .225]
    }
    model_cfg.depth = 4

    model_cfg.input_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    model_cfg.n_lut=cfg.n_lut

    model = CDTNet(depth=4, ch=32, image_fusion=True, attention_mid_k=0.5,
        attend_from=2, batchnorm_from=2, n_lut=cfg.n_lut)
    model.set_resolution(cfg.hr_w, cfg.hr_h, cfg.lr, cfg.finetune_base)
    model.is_sim = cfg.is_sim
    model.to(cfg.device)
    model.encoder.apply(initializer.XavierGluon(rnd_type='gaussian', magnitude=1.0))
    model.decoder.apply(initializer.XavierGluon(rnd_type='gaussian', magnitude=1.0))
    model.lut.classifier.apply(weights_init_normal_classifier)
    torch.nn.init.constant_(model.lut.classifier.fc.bias.data, 1.0)
    model.refine.apply(initializer.XavierGluon(rnd_type='gaussian', magnitude=2.0))

    return model, model_cfg


def model_train(model, cfg, model_cfg):
    cfg.batch_size = 16 if cfg.batch_size < 1 else cfg.batch_size
    cfg.val_batch_size = 1

    cfg.input_normalization = None #model_cfg.input_normalization
    crop_size = model_cfg.crop_size

    loss_cfg = edict()
    loss_cfg.pixel_loss = MaskWeight_MSE(min_area=100)
    loss_cfg.pixel_loss_weight = 1.0
    loss_cfg.base_pixel_loss = MaskWeight_MSE(min_area=100)
    loss_cfg.base_pixel_loss_weight = 1.0
    loss_cfg.lut_loss = MaskWeight_MSE(min_area=100)
    loss_cfg.lut_loss_weight = 1.0

    num_epochs = 120

    train_augmentator = HCompose([
        RandomResizedCrop(*crop_size, scale=(0.5, 1.0)),
        HorizontalFlip(),
    ])

    val_augmentator = HCompose([
        Resize(*crop_size)
    ])

    datasets_names = cfg.datasets.split(',')
    train_datasets_list = []
    if 'HDay2Night' in datasets_names:
        train_datasets_list.append(HDataset(cfg.HDAY2NIGHT_PATH, split='train'))
    if 'HFlickr' in datasets_names:
        train_datasets_list.append(HDataset(cfg.HFLICKR_PATH, split='train'))
    if 'HCOCO' in datasets_names:
        train_datasets_list.append(HDataset(cfg.HCOCO_PATH, split='train'))
    if 'HAdobe5k' in datasets_names:
        train_datasets_list.append(HDataset(cfg.HADOBE5K_PATH, split='train'))
    val_datasets_list = []
    if 'HDay2Night' in datasets_names:
        val_datasets_list.append(HDataset(cfg.HDAY2NIGHT_PATH, split='test'))
    if 'HFlickr' in datasets_names:
        val_datasets_list.append(HDataset(cfg.HFLICKR_PATH, split='test'))
    if 'HCOCO' in datasets_names:
        val_datasets_list.append(HDataset(cfg.HCOCO_PATH, split='test'))
    if 'HAdobe5k' in datasets_names:
        val_datasets_list.append(HDataset(cfg.HADOBE5K_PATH, split='test'))

    trainset = ComposeDataset(
        train_datasets_list,
        augmentator=train_augmentator,
        input_transform=model_cfg.input_transform,
        keep_background_prob=0.05,
    )

    valset = ComposeDataset(
        val_datasets_list,
        augmentator=val_augmentator,
        input_transform=model_cfg.input_transform,
        keep_background_prob=-1,
    )

    optimizer_params = {
        'lr': 1e-3,
        'betas': (0.9, 0.999), 'eps': 1e-8
    }

    lr_scheduler = partial(torch.optim.lr_scheduler.MultiStepLR,
                           milestones=[50, 100], gamma=0.1)
    trainer = SimpleHTrainer(
        model, cfg, model_cfg, loss_cfg,
        trainset, valset,
        optimizer='adam',
        optimizer_params=optimizer_params,
        lr_scheduler=lr_scheduler,
        metrics=[
            PSNRMetric(
                'images', 'target_images'),
            MSEMetric(
                'images', 'target_images')
        ],
        checkpoint_interval=1,
        image_dump_interval=100
    )

    logger.info(f'Starting Epoch: {cfg.start_epoch}')
    logger.info(f'Total Epochs: {num_epochs}')
    for epoch in range(cfg.start_epoch, num_epochs):
        trainer.training(epoch)
        trainer.validation(epoch)
        #trainer.training(epoch)
