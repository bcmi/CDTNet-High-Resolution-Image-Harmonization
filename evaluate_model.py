import argparse
import sys
import os
sys.path.insert(0, os.getcwd())
import torch
from pathlib import Path
from albumentations import Resize, NoOp
from iharm.data.hdataset import HDataset
from iharm.data.transforms import HCompose
from iharm.inference.predictor import Predictor
from iharm.inference.evaluation import evaluate_dataset
from iharm.inference.metrics import MetricsHub, MSE, fMSE, PSNR, fPSNR, SSIM, fSSIM, N
from iharm.inference.utils import load_model, find_checkpoint
from iharm.mconfigs import ALL_MCONFIGS
import yaml
from easydict import EasyDict as edict
from iharm.utils.log import logger, add_new_file_output_to_logger

def load_config_file(config_path, model_name=None, return_edict=False):
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    if 'SUBCONFIGS' in cfg:
        if model_name is not None and model_name in cfg['SUBCONFIGS']:
            cfg.update(cfg['SUBCONFIGS'][model_name])
        del cfg['SUBCONFIGS']

    return edict(cfg) if return_edict else cfg

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('model_type', choices=ALL_MCONFIGS.keys())
    parser.add_argument('checkpoint', type=str,
                        help='The path to the checkpoint. '
                             'This can be a relative path (relative to cfg.MODELS_PATH) '
                             'or an absolute path. The file extension can be omitted.')
    parser.add_argument('--lr', type=int, default=256, help='target base resolution')
    parser.add_argument('--hr', type=int, default=1024, help='target base resolution')
    parser.add_argument('--save_dir', type=str, default='',
                        help='The path to the checkpoint. '
                             'This can be a relative path (relative to cfg.MODELS_PATH) '
                             'or an absolute path. The file extension can be omitted.')
    parser.add_argument('--datasets', type=str, default='HAdobe5K',
                        help='Each dataset name must be one of the prefixes in config paths, '
                             'which look like DATASET_PATH.')
    parser.add_argument('--use-flip', action='store_true', default=False,
                        help='Use horizontal flip test-time augmentation.')
    parser.add_argument('--gpu', type=str, default=0, help='ID of used GPU.')
    parser.add_argument('--config-path', type=str, default='./config.yml',
                        help='The path to the config file.')

    parser.add_argument('--eval-prefix', type=str, default='')

    args = parser.parse_args()
    cfg = load_config_file(args.config_path, return_edict=True)
    return args, cfg


def main():
    args, cfg = parse_args()
    checkpoint_path = find_checkpoint(cfg.MODELS_PATH, args.checkpoint)
    add_new_file_output_to_logger(
        logs_path=Path(cfg.EXPS_PATH) / 'evaluation_logs',
        prefix=f'{Path(checkpoint_path).stem}_',
        only_message=True
    )
    logger.info(vars(args))

    device = torch.device(f'cuda:{args.gpu}')
    net = load_model(args.model_type, checkpoint_path, verbose=True)
    print(net)
    net.set_resolution(args.hr, args.lr, False)
    predictor = Predictor(net, device, with_flip=args.use_flip)
    save_dir = args.save_dir
    if save_dir!='': 
        print(save_dir)
        if not os.path.exists(save_dir): os.makedirs(save_dir)


    datasets_names = args.datasets.split(',')
    datasets_metrics = []
    for dataset_indx, dataset_name in enumerate(datasets_names):
        dataset = HDataset(
            cfg.get(f'{dataset_name.upper()}_PATH'), split='test',
            augmentator=HCompose([Resize(args.hr, args.hr)]),
            keep_background_prob=-1
        )

        dataset_metrics = MetricsHub([MSE(), fMSE(), PSNR(), SSIM()],
                                     name=dataset_name)

        evaluate_dataset(dataset, predictor, dataset_metrics, logger, save_dir)
        datasets_metrics.append(dataset_metrics)
        if dataset_indx == 0:
            logger.info(dataset_metrics.get_table_header())
        logger.info(dataset_metrics)

    if len(datasets_metrics) > 1:
        overall_metrics = sum(datasets_metrics, MetricsHub([], 'Overall'))
        logger.info('-' * len(str(overall_metrics)))
        logger.info(overall_metrics)


if __name__ == '__main__':
    main()
