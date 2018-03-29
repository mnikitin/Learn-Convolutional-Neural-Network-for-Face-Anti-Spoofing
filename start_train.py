import logging, sys, os
import json

from config import config
from utils.net_train import train_net
from utils.data_utils import *

def save_config(cfg):
    save_fname = cfg['experiment_dir'] + '/config.json'
    with open(save_fname, 'w') as f:
        json.dump(cfg, f, sort_keys=True, indent=4)

def set_logging(experiment_dir):
    # duplicate logging to file and stdout
    logging.basicConfig(level=logging.INFO,
                        format='[%(asctime)s]\t%(message)s',
                        datefmt='%m-%d %H:%M',
                        filename=experiment_dir + '/log.txt',
                        filemode='w')
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    logging.getLogger().addHandler(console)

def start_experiment(cfg):
    if not os.path.exists(cfg['experiment_dir']):
        os.makedirs(cfg['experiment_dir'])
    save_config(cfg)
    set_logging(cfg['experiment_dir'])
    train_net(cfg)


def train_net_casia(data_dir, scale, k):
    client_num = 20
    folds_num = 5
    client_per_fold = client_num / folds_num
    list_dir = data_dir + '/mxnet'
    db_dir = data_dir + '/scale_' + str(scale) + '/train_release'

    experiment_str = 'scale_' + str(scale) + '__fold_' + str(k)
    # prepare dev data
    client_list_dev = [i for i in range((k - 1) * client_per_fold + 1, k * client_per_fold + 1)]
    dev_list_fname = list_dir + '/' + experiment_str + '__dev'
    create_imglist_casia(db_dir, dev_list_fname, False, client_list_dev)
    create_record_file(config['mxnet_path'], config['cwd_path'], dev_list_fname)
    # prepare train data
    client_list_train = list(set(range(1, client_num + 1)) - set(client_list_dev))
    train_list_fname = list_dir + '/' + experiment_str + '__train'
    create_imglist_casia(db_dir, train_list_fname, True, client_list_train)
    create_record_file(config['mxnet_path'], config['cwd_path'], train_list_fname)
    # set config
    config['train_db'] = train_list_fname + '.rec'
    config['val_db'] = dev_list_fname + '.rec'
    with open('%s/mean_std__scale_%s.txt' % (data_dir, str(scale)), 'r') as f:
        config['data_params']['mean'] = map(float, f.readline().split())
    config['experiment_dir'] = 'experiments/casia/' + experiment_str
    # start experiment
    start_experiment(config)


def main(argc, argv):
    train_net_casia('data/casia', float(argv[1]), int(argv[2]))


if __name__ == "__main__":
    main(len(sys.argv), sys.argv)
