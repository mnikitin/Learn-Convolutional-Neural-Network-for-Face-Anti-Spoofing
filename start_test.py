import sys, os, json, cv2
from glob import glob
import numpy as np
import mxnet as mx
from collections import namedtuple
Batch = namedtuple('Batch', ['data'])

from svmutil import *
from utils.statistics import get_eer_stats, get_hter_at_thr

def load_net(experiment_dir):
    config_fname = experiment_dir + '/config.json'
    with open(config_fname, 'r') as config_file:
        config = json.load(config_file)
    context = mx.gpu()
    shape = config['input_shape']
    mean = config['data_params']['mean']
    net_output = 'pooling2_output'
    sym, arg_params, aux_params = mx.model.load_checkpoint(experiment_dir + '/net', config['num_epoch'])
    output_sym = sym.get_internals()[net_output]
    net = mx.mod.Module(symbol=output_sym, context=context, label_names=None)
    net.bind(for_training=False, data_shapes=[('data', (1, shape[0], shape[1], shape[2]))])
    net.set_params(arg_params, aux_params)
    return net, mean

def get_cnn_features(img_name, net, mean):
    img = cv2.imread(img_name)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(float)
    img[:,:,0] -= mean[0]
    img[:,:,1] -= mean[1]
    img[:,:,2] -= mean[2]
    img = np.swapaxes(img, 0, 2)
    img = np.swapaxes(img, 1, 2)
    img = img[np.newaxis, :]
    net.forward(Batch([mx.nd.array(img)]))
    features = net.get_outputs()[0][0].asnumpy().flatten().tolist()
    return features

def prepare_svm_data(db_dir, client_list, net, mean):
    names_real = ['1', '2', 'HR_1']
    names_fake = ['3', '4', '5', '6', '7', '8', 'HR_2', 'HR_3', 'HR_4']
    data = []
    labels = []
    for client in client_list:
        for video_name in names_real:
            for frame_name in glob('%s/%d/%s/*.jpg' % (db_dir, client, video_name)):
                data.append(get_cnn_features(frame_name, net, mean))
                labels.append(1)
        for video_name in names_fake:
            for frame_name in glob('%s/%d/%s/*.jpg' % (db_dir, client, video_name)):
                data.append(get_cnn_features(frame_name, net, mean))
                labels.append(-1)
    return data, labels


def train_classifier(data, labels, c):
    prob  = svm_problem(labels, data)
    param = svm_parameter('-t 0 -c %f -b 1 -q' % c)
    model = svm_train(prob, param)
    return model

def estimate_eer(data, labels, model):
    p_label, p_acc, p_val = svm_predict(labels, data, model, '-b 1 -q')
    scores = np.asarray([prob[0] for prob in p_val])
    labels = np.asarray(labels)
    eer, thr = get_eer_stats(scores, labels)
    return eer, thr

def estimate_hter(data, labels, model, thr):
    p_label, p_acc, p_val = svm_predict(labels, data, model, '-b 1 -q')
    scores = np.asarray([prob[0] for prob in p_val])
    labels = np.asarray(labels)
    hter = get_hter_at_thr(scores, labels, thr)
    return hter


def save_results(scale, dev_eer, test_hter, test_eer, svm_c):
    if not os.path.exists('results'):
        os.makedirs('results')
    with open('results/casia__scale_' + str(scale) + '.txt', 'w') as f:
        for c in svm_c:
            f.write('C = %f:\n' % c)
            f.write('\tEER-dev:\t%f (%f, %f, %f, %f, %f)\n' % (np.mean(dev_eer[c]), dev_eer[c][0], dev_eer[c][1], dev_eer[c][2], dev_eer[c][3], dev_eer[c][4]))
            f.write('\tHTER-test:\t%f (%f, %f, %f, %f, %f)\n' % (np.mean(test_hter[c]), test_hter[c][0], test_hter[c][1], test_hter[c][2], test_hter[c][3], test_hter[c][4]))
            f.write('\tEER-test:\t%f (%f, %f, %f, %f, %f)\n' % (np.mean(test_eer[c]), test_eer[c][0], test_eer[c][1], test_eer[c][2], test_eer[c][3], test_eer[c][4]))

def print_results(scale, dev_eer, test_hter, test_eer, svm_c):
    print('CASIA: scale_' + str(scale) + ' results:')
    for c in svm_c:
        print('C = %f:' % c)
        print('\tEER-dev:\t%f (%f, %f, %f, %f, %f)' % (np.mean(dev_eer[c]), dev_eer[c][0], dev_eer[c][1], dev_eer[c][2], dev_eer[c][3], dev_eer[c][4]))
        print('\tHTER-test:\t%f (%f, %f, %f, %f, %f)' % (np.mean(test_hter[c]), test_hter[c][0], test_hter[c][1], test_hter[c][2], test_hter[c][3], test_hter[c][4]))
        print('\tEER-test:\t%f (%f, %f, %f, %f, %f)' % (np.mean(test_eer[c]), test_eer[c][0], test_eer[c][1], test_eer[c][2], test_eer[c][3], test_eer[c][4]))


def test_net_casia(data_dir, experiments_dir, folds_num, svm_c):
    client_num = 20
    client_per_fold = client_num / folds_num

    scales = [1.0, 1.4, 1.8, 2.2, 2.6]
    for scale in scales:
        dev_eer, test_hter, test_eer = {}, {}, {}
        for c in svm_c:
            dev_eer[c] = []
            test_hter[c] = []
            test_eer[c] = []
        db_dir = data_dir + '/scale_' + str(scale)
        for k in range(1, folds_num + 1):
            experiment_str = 'scale_' + str(scale) + '__fold_' + str(k)
            net, mean = load_net(experiments_dir + '/' + experiment_str)
            # prepare client lists
            client_list_dev = [i for i in range((k - 1) * client_per_fold + 1, k * client_per_fold + 1)]
            client_list_train = list(set(range(1, client_num + 1)) - set(client_list_dev))
            client_list_test = [i for i in range(1, 31)]
            # prepare features and labels for svm
            print("Scale: %.1f; Fold: %d. Preparing CNN features ..." % (scale, k))
            data_dev, labels_dev = prepare_svm_data(db_dir + '/train_release', client_list_dev, net, mean)
            data_train, labels_train = prepare_svm_data(db_dir + '/train_release', client_list_train, net, mean)
            data_test, labels_test = prepare_svm_data(db_dir + '/test_release', client_list_test, net, mean)
            # train model and estimate quality
            for c in svm_c:
                print("Scale: %.1f; Fold: %d. Training and estimation SVM with C=%f ..." % (scale, k, c))
                model = train_classifier(data_train, labels_train, c)
                cur_eer_dev, thr = estimate_eer(data_dev, labels_dev, model)
                cur_hter_test = estimate_hter(data_test, labels_test, model, thr)
                cur_eer_test, _ = estimate_eer(data_test, labels_test, model)
                dev_eer[c].append(cur_eer_dev)
                test_hter[c].append(cur_hter_test)
                test_eer[c].append(cur_eer_test)
        # save and print results
        save_results(scale, dev_eer, test_hter, test_eer, svm_c)
        print_results(scale, dev_eer, test_hter, test_eer, svm_c)


def main(argc, argv):
    svm_c = [2 ** i for i in range(-10, 11, 2)]
    test_net_casia('data/casia', 'experiments/casia', 5, svm_c)


if __name__ == "__main__":
    main(len(sys.argv), sys.argv)
