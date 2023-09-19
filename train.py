import argparse
import importlib.util as util

import torch
from iharm.utils.exp import init_experiment
import os


def parse_args():
    '''
    use parser to get command line parameters
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', type=str,
                        help='the model\'s file')

    parser.add_argument('--exp_name', type=str, default='',
                        help='Here you can specify the name of the experiment. '
                             'It will be added as a suffix to the experiment folder.')

    parser.add_argument('--datasets', type=str, default='HDay2Night,HFlickr,HCOCO,HAdobe5k',
                        help='Each dataset name must be one of the prefixes in config paths, '
                             'which look like DATASET_PATH.')

    parser.add_argument('--workers', type=int, default=10,
                        metavar='N', help='Dataloader threads.')

    parser.add_argument('--batch_size', type=int, default=1,
                        help='The batch size while training')
    
    parser.add_argument('--hr_h', type=int, default=1024, help='target h resolution')
    parser.add_argument('--hr_w', type=int, default=1024, help='target w resolution')
    parser.add_argument('--lr', type=int, default=256, help='target base resolution')

    parser.add_argument('--is_sim', action='store_true', default=False,
                        help='Whether use CDTNet-sim.')

    parser.add_argument('--ngpus', type=int, default=1,
                        help='Number of GPUs. '
                             'If you only specify "--gpus" argument, the ngpus value will be calculated automatically. '
                             'You should use either this argument or "--gpus".')

    parser.add_argument('--gpus', type=str, default='', required=False,
                        help='Ids of used GPUs. You should use either this argument or "--ngpus".')

    parser.add_argument('--resume-exp', type=str, default=None,
                        help='The prefix of the name of the experiment to be continued. '
                             'If you use this field, you must specify the "--resume-prefix" argument.')

    parser.add_argument('--resume-prefix', type=str, default='latest',
                        help='The prefix of the name of the checkpoint to be loaded.')

    parser.add_argument('--start_epoch', type=int, default=0,
                        help='The number of the starting epoch from which training will continue. '
                             '(it is important for correct logging and learning rate)')

    parser.add_argument('--weights', type=str, default=None,
                        help='Model weights will be loaded from the specified path if you use this argument.')
                        
    parser.add_argument('--finetune_base', action='store_true',
                        help='Whether finetune the base model')
                        
    parser.add_argument('--n_lut', type=int, default=4)

    return parser.parse_args()


def find_module_name(model_path):
    '''
    According to the parameters, find the location of the script
    '''
    location = util.spec_from_file_location("model_script", model_path)
    model_name = util.module_from_spec(location)
    location.loader.exec_module(model_name)

    return model_name


if __name__ == '__main__':
    #set some flags
    #os.environ["CUDA_VISIBLE_DEVICES"] = '2'
    #torch.cuda.set_device(1)
    torch.backends.cudnn.benchmark = True
    torch.multiprocessing.set_sharing_strategy('file_system')
    #get all configurations from command line and file
    args = parse_args()
    model_name = find_module_name(args.model_path)
    cfg = init_experiment(args)

    #start training!
    model_name.train(cfg)


