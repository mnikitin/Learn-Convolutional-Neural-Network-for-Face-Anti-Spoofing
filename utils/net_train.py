import mxnet as mx

import net_generator
from net_utils import *

def train_net(cfg):
    net = net_generator.get_alexnet(cfg['num_classes'])
    train_iter, val_iter = get_rec_data_iterators(cfg['train_db'], cfg['val_db'], 
                                                  cfg['input_shape'], cfg['batch_size'], cfg['data_params'])

    lr_sch = mx.lr_scheduler.FactorScheduler(cfg['lr_step'], cfg['lr_factor'])
    wd_mult, idx2name = get_wd_mult(net, 1.0)
    optimizer = mx.optimizer.SGD(sym=net, param_idx2name=idx2name, rescale_grad=1.0/cfg['batch_size'], 
                                 momentum=cfg['momentum'], learning_rate=cfg['lr_base'], lr_scheduler=lr_sch, wd=cfg['wd'])
    optimizer.set_wd_mult(wd_mult)

    eval_metrics = [mx.metric.CrossEntropy(),
                    mx.metric.Accuracy()]

    batch_end_callback = mx.callback.Speedometer(cfg['batch_size'], cfg['display'])
    epoch_end_callback = mx.callback.do_checkpoint(cfg['experiment_dir'] + '/' + cfg['model_prefix'])

    train_iter_resized = mx.io.ResizeIter(train_iter, cfg['epoch_size'], reset_internal=False)

    devices = [mx.gpu(device_id) for device_id in cfg['devices_id']]
    net_model = mx.mod.Module(symbol=net, context=devices)

    if cfg['init_from_trained']:
        sym, arg_params, aux_params = mx.model.load_checkpoint(cfg['init_model'][0], cfg['init_model'][1])
        allow_missing = True
        net_model.fit(train_iter_resized,
                      eval_data=val_iter,
                      optimizer=optimizer,
                      eval_metric=eval_metrics,
                      batch_end_callback = batch_end_callback,
                      num_epoch=cfg['num_epoch'],
                      epoch_end_callback = epoch_end_callback,
                      arg_params=arg_params,
                      aux_params=aux_params,
                      allow_missing=allow_missing)
    else:
        initializer = mx.init.MSRAPrelu()
        net_model.fit(train_iter_resized,
                      eval_data=val_iter,
                      initializer=initializer,
                      optimizer=optimizer,
                      eval_metric=eval_metrics,
                      batch_end_callback = batch_end_callback,
                      num_epoch=cfg['num_epoch'],
                      epoch_end_callback = epoch_end_callback)

