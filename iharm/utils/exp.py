import os
import sys
import shutil
import pprint
from pathlib import Path
from datetime import datetime

import yaml
import torch
from easydict import EasyDict as edict

from .log import logger, add_new_file_output_to_logger

def get_filename_list(model_rootpath):
    '''
    Convert the file path into a list
    '''
    file_list = []
    model_name = model_rootpath.stem
    file_list.append(model_name)
    for x in model_rootpath.parents:
        #add to list until the path called 'models'
        if x.stem == 'models':
            break
        file_list.append(x.stem)
    return file_list[::-1]


def load_config_from_file(model_rootpath):
    '''
    give the file's root,load all configuration files
    '''
    model_name = model_rootpath.stem
    configfile_path = model_rootpath.parent / (model_name + '.yml')

    #read the file and add configuration
    if configfile_path.exists():
        with open(configfile_path, 'r') as f:
            new_cfg = yaml.safe_load(f)

        if 'SUBCONFIGS' in new_cfg:
            if model_name is not None and model_name in new_cfg['SUBCONFIGS']:
                new_cfg.update(new_cfg['SUBCONFIGS'][model_name])
            del new_cfg['SUBCONFIGS']
    else:
        new_cfg = dict()

    #if has more configuration files,read them all
    cwd = Path.cwd()
    config_parent_path = configfile_path.parent.absolute()
    while len(config_parent_path.parents) > 0:
        new_configure_path = config_parent_path / 'config.yml'
        if new_configure_path.exists():
            with open(new_configure_path, 'r') as f:
                new_config = yaml.safe_load(f)

            if 'SUBCONFIGS' in new_config:
                if model_name is not None and model_name in new_config['SUBCONFIGS']:
                    new_config.update(new_config['SUBCONFIGS'][model_name])
                del new_config['SUBCONFIGS']
            new_cfg.update({k: v for k, v in new_config.items() if k not in new_cfg})

        if config_parent_path.absolute() == cwd:
            break
        config_parent_path = config_parent_path.parent

    return edict(new_cfg)


def init_experiment(args):
    '''
    #get all configurations from file(filename is in the args)
    '''

    #from file
    model_rootpath = Path(args.model_path)
    filename_list = get_filename_list(model_rootpath)
    cfg = load_config_from_file(model_rootpath)
    #from args
    for param_name, value in vars(args).items():
        if param_name.lower() in cfg or param_name.upper() in cfg:
            continue
        cfg[param_name] = value

    #determine the storage location of the training content according to whether it is to continue training
    exps_parent_path = Path(cfg.EXPS_PATH) / '/'.join(filename_list)
    exps_parent_path.mkdir(parents=True, exist_ok=True)
    if cfg.resume_exp:
        sorted_exps_parent_path = sorted(exps_parent_path.glob(f'{cfg.resume_exp}*'))
        experiment_path = sorted_exps_parent_path[0]
        print(f'Continue with experiment "{experiment_path}"')
    else:
        last_experiment_indx = 0
        for x in exps_parent_path.iterdir():
            if not x.is_dir():
                continue

            exp_name = x.stem
            if exp_name[:3].isnumeric():
                last_experiment_indx = max(last_experiment_indx, int(exp_name[:3]) + 1)

        exp_name = f'{last_experiment_indx:03d}'
        if cfg.exp_name:
            exp_name += '_' + cfg.exp_name
        experiment_path = exps_parent_path / exp_name
        experiment_path.mkdir(parents=True)

    #create parameters for some subdirectories about the content of this experiment
    cfg.EXP_PATH = experiment_path
    cfg.CHECKPOINTS_PATH = experiment_path / 'checkpoints'
    cfg.CHECKPOINTS_PATH.mkdir(exist_ok=True)
    cfg.LOGS_PATH = experiment_path / 'logs'
    cfg.LOGS_PATH.mkdir(exist_ok=True)
    cfg.VIS_PATH = experiment_path / 'vis'
    cfg.VIS_PATH.mkdir(exist_ok=True)

    dst_script_path = experiment_path / (model_rootpath.stem + datetime.strftime(datetime.today(), '_%Y-%m-%d-%H-%M-%S.py'))
    shutil.copy(model_rootpath, dst_script_path)

    #create parameters related to multiple GPUs
    if cfg.gpus != '':
        gpu_ids = [int(gpu_id) for gpu_id in cfg.gpus.split(',')]
    else:
        gpu_ids = list(range(cfg.ngpus))
        cfg.gpus = ','.join([str(id) for id in gpu_ids])
    cfg.multi_gpu = cfg.ngpus > 1
    cfg.gpu_ids = gpu_ids
    cfg.ngpus = len(gpu_ids)
    if cfg.multi_gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = cfg.gpus
    cfg.device = torch.device(f'cuda:{cfg.gpu_ids[0]}')
    # cfg.device = torch.cuda.set_device(1)
    # print('exp',cfg.device)
    # cfg.device=None

    #create a logger
    add_new_file_output_to_logger(cfg.LOGS_PATH, prefix='train_')
    logger.info(f'Number of GPUs: {len(cfg.gpu_ids)}')
    logger.info('Training config:')
    logger.info(pprint.pformat(cfg, indent=4))

    return cfg




