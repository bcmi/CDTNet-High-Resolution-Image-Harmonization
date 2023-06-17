import argparse
import os
import os.path as osp
from pathlib import Path
import sys

import cv2
import numpy as np
import torch
from tqdm import tqdm
import yaml
from easydict import EasyDict as edict

sys.path.insert(0, '.')
from iharm.inference.predictor import Predictor
from iharm.inference.utils import load_model, find_checkpoint
from iharm.mconfigs import ALL_MCONFIGS
from iharm.utils.log import logger

def load_config_file(config_path, model_name=None, return_edict=False):
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    if 'SUBCONFIGS' in cfg:
        if model_name is not None and model_name in cfg['SUBCONFIGS']:
            cfg.update(cfg['SUBCONFIGS'][model_name])
        del cfg['SUBCONFIGS']

    return edict(cfg) if return_edict else cfg

def main():
    args, cfg = parse_args()

    device = torch.device(f'cuda:{args.gpu}')
    checkpoint_path = find_checkpoint(cfg.MODELS_PATH, args.checkpoint)
    net = load_model(args.model_type, checkpoint_path, verbose=True)
    predictor = Predictor(net, device)

    image_names = os.listdir(args.images)

    def _save_image(image_name, bgr_image):
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(
            str(cfg.RESULTS_PATH / f'{image_name}'),
            rgb_image,
            [cv2.IMWRITE_JPEG_QUALITY, 85]
        )

    logger.info(f'Save images to {cfg.RESULTS_PATH}')

    for image_name in tqdm(image_names):
        image_path = osp.join(args.images, image_name)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_size = image.shape
        image = cv2.resize(image, (1024,1024), cv2.INTER_LINEAR)

        mask_path = osp.join(args.masks, image_name)
        print(mask_path)
        mask_image = cv2.imread(mask_path)
        mask_image = cv2.resize(mask_image, (1024,1024), cv2.INTER_LINEAR)
        mask = mask_image[:, :, 0]
        mask[mask <= 100] = 0
        mask[mask > 100] = 1
        mask = mask.astype(np.float32)

        pred = predictor.predict(image, mask)

        if args.original_size:
            pred = cv2.resize(pred, image_size[:-1][::-1])
        _save_image(image_name, pred)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_type', choices=ALL_MCONFIGS.keys())
    parser.add_argument('checkpoint', type=str,
                        help='The path to the checkpoint. '
                             'This can be a relative path (relative to cfg.MODELS_PATH) '
                             'or an absolute path. The file extension can be omitted.')
    parser.add_argument(
        '--images', type=str,
        help='Path to directory with .jpg images to get predictions for.'
    )
    parser.add_argument(
        '--masks', type=str,
        help='Path to directory with .png binary masks for images, named exactly like images without last _postfix.'
    )
    parser.add_argument(
        '--original-size', action='store_true', default=False,
        help='Resize predicted image back to the original size.'
    )
    parser.add_argument('--gpu', type=str, default=0, help='ID of used GPU.')
    parser.add_argument('--config-path', type=str, default='./config.yml', help='The path to the config file.')
    parser.add_argument(
        '--results-path', type=str, default='',
        help='The path to the harmonized images. Default path: cfg.EXPS_PATH/predictions.'
    )

    args = parser.parse_args()
    cfg = load_config_file(args.config_path, return_edict=True)
    cfg.EXPS_PATH = Path(cfg.EXPS_PATH)
    cfg.RESULTS_PATH = Path(args.results_path) if len(args.results_path) else cfg.EXPS_PATH / 'predictions'
    cfg.RESULTS_PATH.mkdir(parents=True, exist_ok=True)
    logger.info(cfg)
    return args, cfg


if __name__ == '__main__':
    main()
