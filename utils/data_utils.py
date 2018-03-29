import os
from glob import glob
import random

def create_record_file(mxnet_path, cwd_path, imglist_fname):
    os.system('python %s/tools/im2rec.py --num-thread=4 --pass-through %s %s' % (mxnet_path, imglist_fname, cwd_path))


def create_imglist_casia(db_dir, save_fname, shuffle, client_list=None):
    names_real = ['1', '2', 'HR_1']
    names_fake = ['3', '4', '5', '6', '7', '8', 'HR_2', 'HR_3', 'HR_4']
    
    if client_list is None:
        client_list = [int(client_dir.split('/')[-1]) for client_dir in glob('%s/*' % db_dir)]

    lines = []
    cnt = 0
    for client in client_list:
        for video_name in names_real:
            for frame_name in glob('%s/%d/%s/*.jpg' % (db_dir, client, video_name)):
                lines.append('%d\t%f\t%s\n' % (cnt, 1.0, frame_name))
                cnt += 1
        for video_name in names_fake:
            for frame_name in glob('%s/%d/%s/*.jpg' % (db_dir, client, video_name)):
                lines.append('%d\t%f\t%s\n' % (cnt, 0.0, frame_name))
                cnt += 1
    if shuffle:
        random.shuffle(lines)

    with open(save_fname + '.lst', 'w') as f:
        f.writelines(lines)


