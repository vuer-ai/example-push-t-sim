import random
import threading
from concurrent.futures import ThreadPoolExecutor

from dotvar import auto_load  # noqa
from pathlib import Path
from textwrap import dedent

import imageio
import numpy as np
import torch
from PIL import Image
from termcolor import cprint
from torch.utils.data import DataLoader
import os

from lucidxr.learning.memmap_cache import dump_memmap, load_memmap
from lucidxr.learning.utils import set_seed
import zarr
import pandas as pd

from vuer_mujoco.schemas.se3.rot_gs6 import gs62quat


class CombinedDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        super(CombinedDataset).__init__()

        self.dataset_collection = datasets
        self.dataset_lens = [len(d) for d in self.dataset_collection]

        # map from global index to dataset index and local index
        arr = []

        for i, dataset_size in enumerate(self.dataset_lens):
            # Iterate through each dataset in the collection and map global indices
            # to corresponding dataset index and item index within each dataset.
            for j in range(dataset_size):
                arr.append((i, j))

        self.dataset_indices = np.array(arr, dtype=np.int64)

    def __len__(self):
        return sum(self.dataset_lens)

    def __getitem__(self, index):
        dataset_idx, local_idx = self.dataset_indices[index]
        return self.dataset_collection[dataset_idx][local_idx]


class EpisodeDataset(torch.utils.data.Dataset):
    max_action_size = 2000

    def __init__(
        self,
        *,
        data_prefix,
        cache_root,
        dataset_host=None,
        episode_paths,
        chunk_size=100,
        image_keys,
        skip_images=False,
        prune_cache=False,
        lucid_mode=False,
        aug_camera_randomization=False,
        use_quat=False,
    ):
        """
        A dataset class for managing episodic data used in machine learning tasks. This dataset handles
        loading, caching, and processing of various types of data such as observations, actions,
        and optionally, image data. The data can either be loaded locally from a cache or pulled remotely.
        """
        super(EpisodeDataset).__init__()
        from ml_logger import ML_Logger

        self.device = None
        self.root_prefix = data_prefix
        self.skip_images = skip_images
        self.use_quat = use_quat
        # self.device = device
        # self.preload_to_device = preload_to_device
        self.cache_root = cache_root
        self.skip_images = skip_images
        self.lucid_mode = lucid_mode
        self.prune_cache = prune_cache
        self.aug_camera_randomization = aug_camera_randomization

        self.chunk_size = chunk_size

        self.cache_prefix = Path(cache_root) / data_prefix.lstrip("/")

        self.image_keys = image_keys
        self.camera_keys = set([x.split("/")[0] for x in self.image_keys])

        self.is_sim = None

        if dataset_host is None:
            dataset_host = os.getenv("ML_LOGGER_HOST")
        if dataset_host.startswith("/"):
            data_prefix = data_prefix.lstrip("/")
        self.loader = ML_Logger(prefix=data_prefix, root=dataset_host)

        # ge: because we pad, fixed episode order is no problem.
        self.episode_paths = sorted(episode_paths)

        data_sizes = []
        def load_episode(i, episode_id):
            print(f"loading episode {episode_id}")
            d = self.load(i, episode_id, load_episodes=True)
            return d["obs"].shape[0] - 1

        with ThreadPoolExecutor(max_workers=16) as executor:
            data_sizes = list(executor.map(lambda args: load_episode(*args), enumerate(self.episode_paths)))
            
        self.data_sizes = data_sizes

    def __getitem__(self, index):

        H = self.chunk_size

        sample_idx, relative_idx = find_ep_id(index, self.data_sizes)
        ep_path = self.episode_paths[sample_idx]

        actions_remaining_in_sample = self.data_sizes[sample_idx] - 1 - relative_idx

        actions_from_next_sample = 0
        pad = False
        if H > actions_remaining_in_sample:
            actions_from_next_sample = H - actions_remaining_in_sample
            H = actions_remaining_in_sample
            pad = True

        load_episodes = False
        data = self.load(sample_idx, ep_path, relative_idx, H, load_episodes=load_episodes)

        if load_episodes:
            obs = data["obs"][relative_idx].float()
            actions = data["actions"][1+relative_idx:1+relative_idx+H].float()
        else:
            obs = data["obs"].float()
            actions = data["actions"].float()
            if self.use_quat:
                xyz = obs[:3]  # shape [3]
                gs6 = obs[3:9]  # shape [6]
                ctrl = obs[9:]  # shape [1]

                quat = gs62quat(gs6)  # shape [4]

                obs = torch.cat([xyz, quat, ctrl], dim=-1)  # shape [8]
                xyz = actions[:, :3]
                gs6 = actions[:, 3:9]
                ctrl = actions[:, 9:]
                quat = gs62quat(gs6)
                actions = torch.cat([xyz, quat, ctrl], dim=-1)  # shape [H, 8]

        all_ep_ids = torch.tensor([False]*actions.shape[0])

        if pad:
            actions = torch.cat((actions, torch.zeros((actions_from_next_sample, actions.shape[1]))), dim=0)
            all_ep_ids = torch.cat((all_ep_ids, torch.full((actions_from_next_sample,), True)), dim=0)

        ep_ids = all_ep_ids != all_ep_ids[0] # dont think this is needed but keeping for now

        batch = {"obs": obs, "actions": actions, "episode_ids": ep_ids, 'episode_index': torch.tensor(sample_idx)}
        if self.device is not None:
            batch = {"obs": obs.to(self.device), "actions": actions.to(self.device), "episode_ids": ep_ids}

        if not self.skip_images:
            for cam_key in self.image_keys:
                image = data[cam_key][relative_idx] if load_episodes else data[cam_key]

                if self.device is not None:
                    image = image.to(self.device)
                cropped_image = image[:, :360, :640] # TODO: hardcoded

                _img = self.preprocess(cropped_image)

                batch[cam_key] = _img

        if self.aug_camera_randomization:
            for k in self.camera_keys:
                if load_episodes:
                    K = data[f"{k}_K"][relative_idx]
                    C2W = data[f"{k}_C2W"][relative_idx]
                else:
                    K = data[f"{k}_K"]
                    C2W = data[f"{k}_C2W"]

                if self.device is not None:
                    K = K.to(self.device)
                    C2W = C2W.to(self.device)

                batch[f"{k}_K"] = K
                batch[f"{k}_C2W"] = C2W

            for k in self.image_keys:
                if load_episodes:
                    depth = data[f"{k}/midas_depth_full"][relative_idx]
                else:
                    depth = data[f"{k}/midas_depth_full"]

                batch[f"{k}/midas_depth_full"] = depth
        return batch


    def load(self, index, episode_key, episode_index=None, H=None, load_episodes=False):
        """

        Args:
            index: the index of the episode
            episode_key: the episode name to load from
            episode_index: the index within the episode to load
            H: the number of actions in the future to load (chunk size)
            load_episodes: if we should load full episodes or not

        Returns:
            a dictionary of the observation, action, episode ids, and images
        """


        prefix = self.cache_prefix / episode_key
        multiview = {}
        if self.prune_cache:
            print("removing the images folder.")
            # Use shutil.rmtree to properly remove the directory and its contents
            for cam_key in self.image_keys:
                key = (prefix / "render" / cam_key).with_suffix(".data")
                try:
                    os.remove(str(key))
                except FileNotFoundError:
                    pass  # Ignore missing file
                except Exception as e:
                    raise e  # Or handle/log it differently if needed

        try:
            if self.prune_cache:
                raise FileNotFoundError("prune_cache is set, so we always load from remote.")
            if load_episodes:
                obs = load_memmap(prefix / "obs.data")
                actions = load_memmap(prefix / "actions.data")

                if self.aug_camera_randomization:
                    for k in self.camera_keys:
                        Ks = load_memmap(prefix / f"{k}_K.data")
                        c2ws = load_memmap(prefix / f"{k}_C2W.data")
                        multiview[f"{k}_K"] = torch.from_numpy(Ks.copy()).float()
                        multiview[f"{k}_C2W"] = torch.from_numpy(c2ws.copy()).float()

            else:
                root = zarr.open_group(str(prefix / "episode_data.zarr"), mode="r")
                obs = root["obs"][episode_index]
                actions = root["actions"][1+episode_index:1+episode_index+H] # make sure you are indexing in multiples of zarr chunk size

                if self.aug_camera_randomization:
                    for k in self.camera_keys:
                        Ks = root[f"{k}_K"][episode_index]
                        c2ws = root[f"{k}_C2W"][episode_index]
                        multiview[f"{k}_K"] = torch.from_numpy(Ks.copy()).float()
                        multiview[f"{k}_C2W"] = torch.from_numpy(c2ws.copy()).float()

            if not self.skip_images:
                for cam_key in self.image_keys:
                    key = (prefix / "render" / cam_key).with_suffix(".data")

                    if load_episodes:
                        single_view = load_memmap(key)
                    else:
                        try:
                            single_view = root[cam_key][episode_index]
                        except Exception as e:
                            print(f"\33[31mError loading {cam_key} from zarr: {e}, episode_index: {episode_key}\033[0m")
                            raise e

                    _image_tensor = torch.from_numpy(single_view.copy())
                    if load_episodes:
                        _image_tensor = _image_tensor.permute(0, 3, 1, 2).contiguous()
                    else:
                        _image_tensor = _image_tensor.permute(2, 0, 1).contiguous()
                    multiview[cam_key] = _image_tensor

                if self.aug_camera_randomization:
                    for k in self.image_keys:
                        cam_key = f"{k}/midas_depth_full"
                        if load_episodes:
                            single_view = load_memmap((prefix / "render" / cam_key).with_suffix(".data"))
                        else:
                            single_view = root[cam_key][episode_index]
                        _image_tensor = torch.from_numpy(single_view.copy())
                        multiview[cam_key] = _image_tensor


        except (FileNotFoundError, KeyError) as e:
            import traceback
            print(os.getcwd())
            print("local cache not found, loading from remote", e)
            traceback.print_exc()
            print("cache:", self.cache_prefix)

            # now load from the remote server.
            stem = Path(episode_key).stem

            obs, actions = self.loader.load_h5(episode_key + ":state,action")

            print("now saving local cache.")
            # save as both memmaps and zarr files
            dump_memmap(obs, prefix / "obs.data")
            dump_memmap(actions, prefix / "actions.data")

            zarr.group(str(prefix / "episode_data.zarr"))

            root = zarr.open_group(str(prefix / "episode_data.zarr"), mode="a")
            zarr_lock = threading.Lock()

            with zarr_lock:
                # set obs and action as zarr chunks
                if 'obs' not in root:
                    obs_arr = root.create_array("obs", shape=obs.shape, chunks=(1, obs.shape[1]), dtype='f4')
                    obs_arr[:] = obs
                if 'actions' not in root:
                    action_arr = root.create_array("actions", shape=actions.shape, chunks=(self.chunk_size, actions.shape[1]), dtype='f4')
                    action_arr[:] = actions

            if self.aug_camera_randomization:
                # load the cameras and depth renders
                for k in self.camera_keys:
                    Ks, c2ws = self.loader.load_h5(episode_key + f":{k}/K,{k}/C2W")
                    dump_memmap(Ks, prefix / f"{k}_K.data")
                    dump_memmap(c2ws, prefix / f"{k}_C2W.data")

                    with zarr_lock:
                        if f"{k}_K" not in root:
                            k_arr = root.create_array(f"{k}_K", shape=Ks.shape, chunks=(self.chunk_size, *Ks.shape[1:]), dtype='f4')
                            k_arr[:] = Ks
                        if f"{k}_C2W" not in root:
                            c2w_arr = root.create_array(f"{k}_C2W", shape=c2ws.shape, chunks=(self.chunk_size, *c2ws.shape[1:]), dtype='f4')
                            c2w_arr[:] = c2ws

            if not load_episodes:
                obs = obs[episode_index]
                actions = actions[1+episode_index:1+episode_index+H]

            def load_cam_video(cam_key):

                session_prefix = Path(episode_key).parent.parent
                with self.loader.Prefix(session_prefix):
                    video_path = f"videos/{stem}/{cam_key}.mp4"
                    video_memory = self.loader.load_file(video_path)
                    print(video_path)
                    try:
                        frames = list(imageio.v3.imiter(video_memory, plugin="pyav"))  # PyAV backend
                    except Exception as e:
                        print(f"\33[31mError loading video {video_path}: {e}\033[0m")
                        raise e
                    if len(frames) != len(obs):
                        print("WARNING: video length does not match observation length!", f"{len(frames)} != {len(obs)}")

                stacked = np.stack(frames, axis=0)
                # tensor = torch.from_numpy(stacked)
                return stacked, len(frames)

            if not self.skip_images:
                for cam_key in self.image_keys:
                    single_view, num_frames = load_cam_video(cam_key)
                    file_key = (prefix / "render" / cam_key).with_suffix(".data")
                    dump_memmap(single_view, file_key)
                    try:
                        with zarr_lock:
                            if cam_key not in root:
                                cam_arr = root.create_array(cam_key, shape=single_view.shape, chunks=(1, 360, 640, 3), dtype='uint8') # TODO: hardcoded shape
                                cam_arr[:] = single_view
                    except Exception as e:
                        print("Another thread is creating the array, ignoring this error.", e)

                    _image_tensor = torch.from_numpy(single_view.copy())
                    multiview[cam_key] = torch.einsum("bhwc->bchw", _image_tensor)

                    if not load_episodes:
                        multiview[cam_key] = multiview[cam_key][episode_index]
                if self.aug_camera_randomization:
                    for k in self.image_keys:
                        cam_key = f"{k}/midas_depth_full"
                        single_view, num_frames = load_cam_video(cam_key)
                        single_view = single_view[..., 0] # should be TxHxW

                        file_key = (prefix / "render" / cam_key).with_suffix(".data")

                        dump_memmap(single_view, file_key)
                        try:
                            with zarr_lock:
                                if cam_key not in root:
                                    cam_arr = root.create_array(cam_key, shape=single_view.shape, chunks=(1, 360, 640), dtype='float32')
                                    cam_arr[:] = single_view
                        except:
                            print("Another thread is creating the array, ignoring this error.")

                        _image_tensor = torch.from_numpy(single_view.copy())
                        multiview[cam_key] = _image_tensor

                        if not load_episodes:
                            multiview[cam_key] = multiview[cam_key][episode_index]

        if load_episodes:
            min_frames = min([view.shape[0] for view in multiview.values()] + [len(obs), len(actions)])
            if min_frames < len(obs):
                print("second time")
                print(f"WARNING: video length does not match observation length! {min_frames} != {len(obs)}")
                print(f"WARNING: truncating the obs {len(obs)} to {min_frames} frames.")
                obs = obs[:min_frames]
                actions = actions[:min_frames]
                for cam_key in self.image_keys:
                    multiview[cam_key] = multiview[cam_key][:min_frames]

                if self.aug_camera_randomization:
                    for k in self.image_keys:
                        cam_key = f"{k}/midas_depth_full"
                        multiview[cam_key] = multiview[cam_key][:min_frames]

        return dict(
            episode_id=torch.LongTensor([index] * len(obs)),
            obs=torch.from_numpy(np.array(obs, copy=True)),
            actions=torch.from_numpy(np.array(actions, copy=True)),
            **multiview,
        )

    def to(self, device):
        self.device = device
        return self

    def __len__(self):
        return sum(self.data_sizes)

    def preprocess(self, image):
        image = image / 255.0
        return image.float()


# TODO this is going to be broken with the lazy episode dataset
class HistoryDataset(EpisodeDataset):
    def __init__(self, *args, history_len, **kwargs):
        super().__init__(*args, **kwargs)
        self.history_len = history_len
        self.frame_indices = [*range(0, -history_len, -1)]
        # self.obs =

    def __getitem__(self, index):
        # sample_full_episode = False  # hardcode
        H = self.chunk_size
        # l_img = self.history_length_image
        l = len(self)

        obs_stack = tuple(self.obs[index + t] for t in self.frame_indices)
        obs = torch.stack(obs_stack, dim=0)

        actions = self.actions[index: index + H]
        # eventually I want to support concatenated episodic data.
        ep_ids = self.episode_ids[index: index + H] != self.episode_ids[index]

        if (index + H) > l:
            actions = torch.cat([actions, torch.zeros((index + H - l, actions.shape[1]))])
            ep_ids = torch.cat([ep_ids, torch.full([index + H - l], True)])

        batch = {"obs": obs, "actions": actions, "episode_ids": ep_ids}

        if not self.skip_images:
            for cam_key in self.image_keys:
                _img = self.preprocess(self.images[cam_key][index])

                batch[cam_key] = _img

        return batch

def load_data_combined(
        dataset_dirs,
        cache_root,
        image_keys,
        batch_size_train,
        batch_size_val,
        train_ratio=0.8,
        debug=False,
        lucid_mode=False,
        aug_camera_randomization=False,
        **kwargs,
):
    from ml_logger import ML_Logger

    trainsets = []
    valsets = []

    loader = ML_Logger()

    for data_path in dataset_dirs[: 1 if debug else None]:
        loader.configure(prefix=data_path)

        all_entries = loader.glob("**/*.h5")
        # all_entries = [entry for entry in all_entries if "00003" in entry or "00004" in entry]
        print(f"Found {len(all_entries)} episodes")
        entries_shuffled = np.random.permutation([*all_entries])
        entries_train = entries_shuffled[: int(train_ratio * len(entries_shuffled))]
        entries_eval = entries_shuffled[int(train_ratio * len(entries_shuffled)):]

        cprint(f"train: {len(entries_train)}, val: {len(entries_eval)}", color="green")

        # construct dataset and dataloader
        train_dataset = EpisodeDataset(
            data_prefix=data_path,
            cache_root=cache_root,
            episode_paths=entries_train[: 1 if debug else None],
            image_keys=image_keys,
            lucid_mode=lucid_mode,
            aug_camera_randomization=aug_camera_randomization,
            **kwargs,
        )
        val_dataset = EpisodeDataset(
            data_prefix=data_path,
            cache_root=cache_root,
            episode_paths=entries_eval[: 1 if debug else None],
            image_keys=image_keys,
            lucid_mode=lucid_mode,
            aug_camera_randomization=aug_camera_randomization,
            **kwargs,
        )

        trainsets.append(train_dataset)
        valsets.append(val_dataset)

    combined_train_dataset = CombinedDataset(*trainsets)
    combined_val_dataset = CombinedDataset(*valsets)

    train_dataloader = DataLoader(
        combined_train_dataset,
        batch_size=batch_size_train,
        shuffle=True,
        pin_memory=False,
        num_workers=0 if debug else 8,
        prefetch_factor=None if debug else 2,
        persistent_workers=not debug,
    )
    val_dataloader = DataLoader(
        combined_val_dataset,
        batch_size=batch_size_val,
        shuffle=True,
        pin_memory=False,
        num_workers=0 if debug else 8,
        prefetch_factor=None if debug else 2,
        persistent_workers=not debug,
    )

    return train_dataloader, val_dataloader


def get_stats(data_loader: DataLoader):
    composite_dataset = data_loader.dataset
    for dset in composite_dataset.dataset_collection:
        stats = {
            "obs:": dict(
                shape=dset.obs.shape,
                mean=dset.obs.mean(0).tolist(),
                std=dset.obs.std(0),
            ),
            "actions:": dict(
                shape=dset.actions.shape,
                mean=dset.actions.mean(0).tolist(),
                std=dset.actions.std(0),
            ),
        }

        if not dset.skip_images:
            for view in dset.image_keys:
                # we subsample since all images are too much. - Ge
                image_values = dset.images[view][::53] / 255.0

                stats[view] = dict(
                    shape=dset.images[view].shape,
                    mean=image_values.mean(dim=(0, 1, 2)).tolist(),
                    std=image_values.std(dim=(0, 1, 2)).tolist(),
                )

        return stats


### env utils

def sample_box_pose():
    x_range = [0.0, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    cube_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    cube_quat = np.array([1, 0, 0, 0])
    return np.concatenate([cube_position, cube_quat])


def sample_insertion_pose():
    # Peg
    x_range = [0.1, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    peg_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    peg_quat = np.array([1, 0, 0, 0])
    peg_pose = np.concatenate([peg_position, peg_quat])

    # Socket
    x_range = [-0.2, -0.1]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    socket_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    socket_quat = np.array([1, 0, 0, 0])
    socket_pose = np.concatenate([socket_position, socket_quat])

    return peg_pose, socket_pose


### helper functions


def compute_dict_mean(epoch_dicts):
    result = {k: None for k in epoch_dicts[0]}
    num_items = len(epoch_dicts)
    for k in result:
        value_sum = 0
        for epoch_dict in epoch_dicts:
            value_sum += epoch_dict[k]
        result[k] = value_sum / num_items
    return result


def detach_dict(d):
    new_d = dict()
    for k, v in d.items():
        new_d[k] = v.detach()
    return new_d

def concat_fast(batch_list, dim=0):
    # works for dim=0; extend for other dims if needed
    first = batch_list[0]
    total = sum(t.shape[dim] for t in batch_list)
    out_shape = (total, *first.shape[1:])
    out = torch.empty(out_shape, dtype=first.dtype)

    offset = 0
    for t in batch_list:
        n = t.shape[dim]
        out[offset:offset+n].copy_(t)
        offset += n
    return out

def find_ep_id(index, data_sizes):
    ep_id = -1
    rel_idx = 0
    for i, length in enumerate(data_sizes):
        index -= length
        if index < 0:
            ep_id = i
            rel_idx = index + length
            break

    assert ep_id >= 0, "Index too large."

    return ep_id, rel_idx

import sys
