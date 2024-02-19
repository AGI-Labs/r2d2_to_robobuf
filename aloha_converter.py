import argparse
import pickle as pkl
from tqdm import tqdm

from typing import Iterator, Tuple, Any
import glob
import numpy as np
import os
import cv2
import h5py
import json
import io
from collections import defaultdict
import random
from copy import deepcopy
from scipy.spatial.transform import Rotation as R


IMAGE_SIZE = (256, 256)
CAM_NAMES = ['cam_high', 'cam_left_wrist', 'cam_low', 'cam_right_wrist']


def crawler(dirname):
    return glob.glob(os.path.join(dirname, '**/*.hdf5'), recursive=True)


def _resize_and_encode(bgr_image, size=IMAGE_SIZE):
    bgr_image = cv2.resize(bgr_image, size, interpolation=cv2.INTER_AREA)
    _, encoded = cv2.imencode(".jpg", bgr_image)
    return encoded


def _decode_bgr_image(img_str):
    return cv2.imdecode(img_str, 1)


def _to_np(grip_value):
    return np.array([grip_value])


def _gaussian_norm(all_acs):
    print('Using gaussian norm')
    all_acs_arr = np.array(all_acs)
    mean = np.mean(all_acs_arr, axis=0)
    std =  np.std(all_acs_arr, axis=0)
    if not std.all(): # handle situation w/ all 0 actions
        std[std == 0] = 1e-17

    for a in all_acs:
        a -= mean
        a /= std

    return dict(loc=mean.tolist(), scale=std.tolist())


def _max_min_norm(all_acs):
    print('Using max min norm')
    all_acs_arr = np.array(all_acs)
    max_ac = np.max(all_acs_arr, axis=0)
    min_ac = np.min(all_acs_arr, axis=0)

    mid = (max_ac + min_ac) / 2
    delta = (max_ac - min_ac) / 2

    for a in all_acs:
        a -= mid
        a /= delta
    return dict(loc=mid.tolist(), scale=delta.tolist())


def convert_dataset(base_path, gaussian_norm):
    print(f'gaussian_norm={gaussian_norm}')
    print()
    
    episode_paths = crawler(base_path)

    out_trajs, all_acs = [], []
    for episode_path in tqdm(episode_paths):
        proc_traj = []
        with h5py.File(episode_path, 'r') as f:
            actions = f['action'][:]
        
            for t, a in enumerate(actions):
                all_acs.append(a) # for normalization later

                reward = 0 # dummy reward
                obs = dict(state=f['observations']['qpos'][t])
                for idx, key in enumerate(CAM_NAMES):
                    bgr_img = _decode_bgr_image(f['observations']['images'][key][t])
                    obs[f'enc_cam_{idx}'] = _resize_and_encode(bgr_img)
                proc_traj.append((obs, a, reward))
        out_trajs.append(proc_traj)

    ac_dict = _max_min_norm(all_acs) if not gaussian_norm \
              else _gaussian_norm(all_acs)

    with open('ac_norm.json', 'w') as f:
        json.dump(ac_dict, f)

    with open('buf.pkl', 'wb') as f:
        pkl.dump(out_trajs, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path')
    parser.add_argument('--gaussian_norm', action='store_true')
    args = parser.parse_args()
    convert_dataset(os.path.expanduser(args.path), args.gaussian_norm)
