import argparse
from robobuf.buffers import ObsWrapper, ReplayBuffer, Transition
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
from PIL import Image
from scipy.spatial.transform import Rotation as R


def euler_to_r6(euler, degrees=False):
    rot_mat = R.from_euler("xyz", euler, degrees=degrees).as_matrix()
    a1, a2 = rot_mat[0], rot_mat[1]
    return np.concatenate((a1, a2))


camera_type_dict = {
    'hand_camera_id': 0,
    'varied_camera_1_id': 1,
    'varied_camera_2_id': 1,
}

camera_type_to_string_dict = {
    0: "hand_camera",
    1: "varied_camera",
    2: "fixed_camera",
}


def get_camera_type(cam_id):
    if cam_id not in camera_type_dict:
        return None
    type_int = camera_type_dict[cam_id]
    type_str = camera_type_to_string_dict[type_int]
    return type_str


class MP4Reader:
    def __init__(self, filepath, serial_number):
        # Save Parameters #
        self.serial_number = serial_number
        self._index = 0

        # Open Video Reader #
        self._mp4_reader = cv2.VideoCapture(filepath)
        if not self._mp4_reader.isOpened():
            raise RuntimeError("Corrupted MP4 File")


    def set_reading_parameters(
        self,
        image=True,
        concatenate_images=False,
        resolution=(0, 0),
        resize_func=None,
    ):
        # Save Parameters #
        self.image = image
        self.concatenate_images = concatenate_images
        self.resolution = resolution
        self.resize_func = cv2.resize
        self.skip_reading = not image
        if self.skip_reading:
            return

    def get_frame_resolution(self):
        width = self._mp4_reader.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)
        height = self._mp4_reader.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)
        return (width, height)

    def get_frame_count(self):
        if self.skip_reading:
            return 0
        frame_count = int(self._mp4_reader.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
        return frame_count

    def set_frame_index(self, index):
        if self.skip_reading:
            return

        if index < self._index:
            self._mp4_reader.set(cv2.CAP_PROP_POS_FRAMES, index - 1)
            self._index = index

        while self._index < index:
            self.read_camera(ignore_data=True)

    def _process_frame(self, frame):
        frame = deepcopy(frame)
        if self.resolution == (0, 0):
            return frame
        return self.resize_func(frame, self.resolution)
        #return cv2.resize(frame, self.resolution, interpolation=cv2.INTER_AREA)

    def read_camera(self, ignore_data=False, correct_timestamp=None):
        # Skip if Read Unnecesary #
        if self.skip_reading:
            return {}

        # Read Camera #
        success, frame = self._mp4_reader.read()

        self._index += 1
        if not success:
            return None
        if ignore_data:
            return None

        # Return Data #
        data_dict = {}

        if self.concatenate_images:
            data_dict["image"] = {self.serial_number: self._process_frame(frame)}
        else:
            single_width = frame.shape[1] // 2
            data_dict["image"] = {
                self.serial_number + "_left": self._process_frame(frame[:, :single_width, :]),
                self.serial_number + "_right": self._process_frame(frame[:, single_width:, :]),
            }

        return data_dict

    def disable_camera(self):
        if hasattr(self, "_mp4_reader"):
            self._mp4_reader.release()


class RecordedMultiCameraWrapper:
    def __init__(self, recording_folderpath, camera_kwargs={}):
        # Save Camera Info #
        self.camera_kwargs = camera_kwargs

        # Open Camera Readers #
        mp4_filepaths = glob.glob(recording_folderpath + "/*.mp4")
        svo_filepaths = []
        all_filepaths = svo_filepaths + mp4_filepaths

        self.camera_dict = {}
        for f in all_filepaths:
            serial_number = f.split("/")[-1][:-4]
            cam_type = get_camera_type(serial_number)
            camera_kwargs.get(cam_type, {})

            if f.endswith(".mp4"):
                Reader = MP4Reader
            else:
                raise ValueError

            self.camera_dict[serial_number] = Reader(f, serial_number)

    def read_cameras(self, index=None, camera_type_dict={}, timestamp_dict={}):
        full_obs_dict = defaultdict(dict)

        # Read Cameras In Randomized Order #
        all_cam_ids = list(self.camera_dict.keys())
        #random.shuffle(all_cam_ids)

        for cam_id in all_cam_ids:
            cam_type = camera_type_dict[cam_id]
            curr_cam_kwargs = self.camera_kwargs.get(cam_type, {})
            self.camera_dict[cam_id].set_reading_parameters(**curr_cam_kwargs)

            timestamp = timestamp_dict.get(cam_id + "_frame_received", None)
            if index is not None:
                self.camera_dict[cam_id].set_frame_index(index)

            data_dict = self.camera_dict[cam_id].read_camera(correct_timestamp=timestamp)

            # Process Returned Data #
            if data_dict is None:
                return None
            for key in data_dict:
                full_obs_dict[key].update(data_dict[key])

        return full_obs_dict



def get_hdf5_length(hdf5_file, keys_to_ignore=[]):
    length = None

    for key in hdf5_file.keys():
        if key in keys_to_ignore:
            continue

        curr_data = hdf5_file[key]
        if isinstance(curr_data, h5py.Group):
            curr_length = get_hdf5_length(curr_data, keys_to_ignore=keys_to_ignore)
        elif isinstance(curr_data, h5py.Dataset):
            curr_length = len(curr_data)
        else:
            raise ValueError

        if length is None:
            length = curr_length
        assert curr_length == length

    return length


def load_hdf5_to_dict(hdf5_file, index, keys_to_ignore=[]):
    data_dict = {}

    for key in hdf5_file.keys():
        if key in keys_to_ignore:
            continue

        curr_data = hdf5_file[key]
        if isinstance(curr_data, h5py.Group):
            data_dict[key] = load_hdf5_to_dict(curr_data, index, keys_to_ignore=keys_to_ignore)
        elif isinstance(curr_data, h5py.Dataset):
            data_dict[key] = curr_data[index]
        else:
            raise ValueError

    return data_dict



class TrajectoryReader:
    def __init__(self, filepath, read_images=True):
        self._hdf5_file = h5py.File(filepath, "r")
        is_video_folder = "observations/videos" in self._hdf5_file
        self._read_images = read_images and is_video_folder
        self._length = get_hdf5_length(self._hdf5_file)
        self._video_readers = {}
        self._index = 0

    def length(self):
        return self._length

    def read_timestep(self, index=None, keys_to_ignore=[]):
        # Make Sure We Read Within Range #
        if index is None:
            index = self._index
        else:
            assert not self._read_images
            self._index = index
        assert index < self._length

        # Load Low Dimensional Data #
        keys_to_ignore = [*keys_to_ignore.copy(), "videos"]
        timestep = load_hdf5_to_dict(self._hdf5_file, self._index, keys_to_ignore=keys_to_ignore)

        # Increment Read Index #
        self._index += 1

        # Return Timestep #
        return timestep

    def close(self):
        self._hdf5_file.close()


def crawler(dirname, filter_func=None):
    subfolders = [f.path for f in os.scandir(dirname) if f.is_dir()]
    traj_files = [f.path for f in os.scandir(dirname) if (f.is_file() and "trajectory.h5" in f.path)]

    if len(traj_files):
        # Only Save Desired Data #
        if filter_func is None:
            use_data = True
        else:
            hdf5_file = h5py.File(traj_files[0], "r")
            use_data = filter_func(hdf5_file.attrs)
            hdf5_file.close()

        if use_data:
            return [dirname]

    all_folderpaths = []
    for child_dirname in subfolders:
        child_paths = crawler(child_dirname, filter_func=filter_func)
        all_folderpaths.extend(child_paths)

    return all_folderpaths


def load_trajectory(
    filepath=None,
    read_cameras=True,
    recording_folderpath=None,
    camera_kwargs={},
    remove_skipped_steps=False,
    num_samples_per_traj=None,
    num_samples_per_traj_coeff=1.5,
):
    read_hdf5_images = read_cameras and (recording_folderpath is None)
    read_recording_folderpath = read_cameras and (recording_folderpath is not None)

    traj_reader = TrajectoryReader(filepath, read_images=read_hdf5_images)
    if read_recording_folderpath:
        camera_reader = RecordedMultiCameraWrapper(recording_folderpath, camera_kwargs)

    horizon = traj_reader.length()
    timestep_list = []

    # Choose Timesteps To Save #
    if num_samples_per_traj:
        num_to_save = num_samples_per_traj
        if remove_skipped_steps:
            num_to_save = int(num_to_save * num_samples_per_traj_coeff)
        max_size = min(num_to_save, horizon)
        indices_to_save = np.sort(np.random.choice(horizon, size=max_size, replace=False))
    else:
        indices_to_save = np.arange(horizon)

    # Iterate Over Trajectory #
    for i in indices_to_save:
        # Get HDF5 Data #
        timestep = traj_reader.read_timestep(index=i)

        # If Applicable, Get Recorded Data #
        if read_recording_folderpath:
            timestamp_dict = timestep["observation"]["timestamp"]["cameras"]
            camera_type_dict = {
                k: camera_type_to_string_dict[v] for k, v in timestep["observation"]["camera_type"].items()
            }
            camera_obs = camera_reader.read_cameras(
                index=i, camera_type_dict=camera_type_dict, timestamp_dict=timestamp_dict
            )
            camera_failed = camera_obs is None

            # Add Data To Timestep If Successful #
            if camera_failed:
                break
            else:
                timestep["observation"].update(camera_obs)
        
        # Filter Steps #
        step_skipped = not timestep["observation"]["controller_info"].get("movement_enabled", True)
        delete_skipped_step = step_skipped and remove_skipped_steps

        # Save Filtered Timesteps #
        if delete_skipped_step:
            del timestep
        else:
            timestep_list.append(timestep)

    # Remove Extra Transitions #
    timestep_list = np.array(timestep_list)
    if (num_samples_per_traj is not None) and (len(timestep_list) > num_samples_per_traj):
        ind_to_keep = np.random.choice(len(timestep_list), size=num_samples_per_traj, replace=False)
        timestep_list = timestep_list[ind_to_keep]

    # Close Readers #
    traj_reader.close()

    # Return Data #
    return timestep_list


def _resize_and_encode(bgr_image, size):
    bgr_image = cv2.resize(bgr_image, size, interpolation=cv2.INTER_AREA)
    _, encoded = cv2.imencode(".jpg", bgr_image)
    return encoded


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


def convert_dataset(base_path, absolute_ac, max_min_norm, use_r6):
    gripper_ac_key = 'gripper_position' if absolute_ac else 'gripper_velocity'
    robot_ac_key = 'cartesian_position' if absolute_ac else 'cartesian_velocity'
    print(f'Using actions {robot_ac_key} and {gripper_ac_key}')
    print(f'max_min_norm={max_min_norm} and use_r6={use_r6}')
    print()
    
    episode_paths = crawler(base_path)
    episode_paths = [p for p in episode_paths if os.path.exists(p + '/trajectory.h5') and \
            os.path.exists(p + '/recordings/MP4')]

    out_trajs, all_acs = [], []
    for episode_path in tqdm(episode_paths):
        FRAMESKIP = 1
        IMAGE_SIZE = (256, 256)

        h5_filepath = os.path.join(episode_path, 'trajectory.h5')
        recording_folderpath = os.path.join(episode_path, 'recordings', 'MP4')

        traj = load_trajectory(h5_filepath, recording_folderpath=recording_folderpath)
        data  = traj[::FRAMESKIP]
        assert all(t.keys() == data[0].keys() for t in data)

        proc_traj, started = [], False
        for t, step in enumerate(data):
            obs_dict = step['observation']
            action_dict = step['action']
            camera_type_dict = obs_dict['camera_type']

            # important to keep order consistent, in case r2d2 flips internally
            wrist_ids = sorted([k for k, v in camera_type_dict.items() if v == 0])
            exterior_ids = sorted([k for k, v in camera_type_dict.items() if v != 0])

            reward = 0  # DUMMY VALUE
            rbt_action = action_dict[robot_ac_key]
            rbt_action = np.concatenate((rbt_action[:3], euler_to_r6(rbt_action[3:]))) if use_r6 \
                         else rbt_action
            action = np.concatenate((rbt_action, _to_np(action_dict[gripper_ac_key]))).astype(np.float32)

            if np.sum(np.abs(action)) == 0 and not started:
                continue
            started = True
            all_acs.append(action)

            # obs dict
            obs = dict()
            obs['state'] = np.concatenate((obs_dict['robot_state']['cartesian_position'], _to_np(obs_dict['robot_state']['gripper_position']))).astype(np.float32)
            for idx, key in enumerate(wrist_ids + exterior_ids):
                obs[f'enc_cam_{idx}'] = _resize_and_encode(obs_dict['image'][f'{key}_left'], IMAGE_SIZE)
            proc_traj.append((obs, action, reward))
        out_trajs.append(proc_traj)

    ac_dict = _max_min_norm(all_acs) if max_min_norm \
              else _gaussian_norm(all_acs)

    with open('ac_norm.json', 'w') as f:
        json.dump(ac_dict, f)

    with open('buf.pkl', 'wb') as f:
        pkl.dump(out_trajs, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path')
    parser.add_argument('--absolute', action='store_true')
    parser.add_argument('--max_min_norm', action='store_true')
    parser.add_argument('--use_r6', action='store_true')
    args = parser.parse_args()
    convert_dataset(os.path.expanduser(args.path), args.absolute, args.max_min_norm, args.use_r6)
