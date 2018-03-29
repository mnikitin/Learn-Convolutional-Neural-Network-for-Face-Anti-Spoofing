import numpy as np
import mxnet as mx

def get_wd_mult(symbol, multiplier):
    internals = symbol.get_internals()
    arg_names = internals.list_arguments()
    wd_dict = dict()
    idx2name = dict()
    for idx, arg_name in enumerate(arg_names):
        wd_dict[arg_name] = multiplier
        idx2name[idx] = arg_name
    return wd_dict, idx2name


def get_raw_data_iterators(path_root, train_imglist, val_imglist, input_shape, batch_size, data_params):
    mean = np.asarray(data_params['mean'])
    train = mx.image.ImageIter(path_root=path_root, path_imglist=train_imglist, 
                               data_shape=input_shape, shuffle=data_params['shuffle'], batch_size=batch_size,
                               mean=mean, rand_crop=data_params['rand_crop'], rand_mirror=data_params['rand_mirror'])
    val = mx.image.ImageIter(path_root=path_root, path_imglist=val_imglist,
                             data_shape=input_shape, batch_size=batch_size,
                             mean=mean)
    return train, val

def get_rec_data_iterators(train_db, val_db, input_shape, batch_size, data_params):
    train = mx.io.ImageRecordIter(path_imgrec=train_db, data_shape=input_shape, batch_size=batch_size, shuffle=data_params['shuffle'],
                                  mean_r=data_params['mean'][0], mean_g=data_params['mean'][1], mean_b=data_params['mean'][2],
                                  rand_crop=data_params['rand_crop'], rand_mirror=data_params['rand_mirror'])
    val = mx.io.ImageRecordIter(path_imgrec=val_db, data_shape=input_shape, batch_size=batch_size,
                                mean_r=data_params['mean'][0], mean_g=data_params['mean'][1], mean_b=data_params['mean'][2])
    return train, val
