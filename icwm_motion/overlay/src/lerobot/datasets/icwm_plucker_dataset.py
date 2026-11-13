"""Attach verified per-frame calibration sidecars without changing source datasets."""
import json
import hashlib
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

class ICWMPluckerCalibratedDataset(Dataset):
    def __init__(self, dataset, calibration_dir, frame='world'):
        self.dataset = dataset
        self.frame = frame
        self.calibrations = []
        datasets = getattr(dataset, 'query_datasets', [dataset])
        self.cumulative = np.cumsum([0]+[len(d) for d in datasets])
        for ds in datasets:
            folder = Path(calibration_dir)/Path(ds.root).name
            meta = json.loads((folder/'calibration.json').read_text())
            if not meta.get('validated') or meta['convention'] != 'opengl' or meta.get('calibration_version') != 'raw_state_replay_v1':
                raise ValueError(f'unvalidated camera calibration: {folder}')
            if Path(meta['dataset_root']).resolve() != Path(ds.root).resolve() or meta['frames'] != len(ds):
                raise ValueError(f'calibration/data alignment mismatch: {ds.root}')
            if hashlib.sha256((Path(ds.root)/'meta/info.json').read_bytes()).hexdigest() != meta['info_sha256']:
                raise ValueError(f'dataset metadata changed after calibration: {ds.root}')
            K = np.load(folder/'intrinsics.npy', mmap_mode='r')
            poses = np.load(folder/'camera_to_world.npy', mmap_mode='r')
            if K.shape != (len(ds), len(meta['image_keys']), 3, 3) or poses.shape != (len(ds), len(meta['image_keys']), 4, 4):
                raise ValueError(f'wrong calibration shapes: {folder}')
            base = None
            if frame == 'base':
                # per-frame robot0_link0 pose in the world (scripts/prepare_robot_base.py), same frame order
                base_meta = json.loads((folder/'robot_base.json').read_text())
                if not base_meta.get('validated') or base_meta['frames'] != len(ds) or base_meta['info_sha256'] != meta['info_sha256']:
                    raise ValueError(f'unvalidated robot-base sidecar: {folder}')
                base = np.load(folder/'robot_base_to_world.npy', mmap_mode='r')
                if base.shape != (len(ds), 4, 4):
                    raise ValueError(f'wrong robot-base sidecar shape: {folder}')
            elif frame != 'world':
                raise ValueError(f'unknown plucker frame {frame}')
            self.calibrations.append((meta, K, poses, base))

    def __len__(self):
        return len(self.dataset)

    def __getattr__(self, name):
        dataset = self.__dict__.get('dataset')
        if dataset is None:
            raise AttributeError(name)
        return getattr(dataset, name)

    def __getitem__(self, index):
        from lerobot.policies.icwm.plucker import ROBOT_BASE_KEY, camera_keys
        k = int(np.searchsorted(self.cumulative, index, side='right')-1)
        local = int(index-self.cumulative[k])
        item = dict(self.dataset[index])
        meta, intrinsics, poses, base = self.calibrations[k]
        if base is not None:
            item[ROBOT_BASE_KEY] = torch.from_numpy(base[local].copy())
        for view, image_key in enumerate(meta['image_keys']):
            k_key, pose_key = camera_keys(image_key)
            item[k_key] = torch.from_numpy(intrinsics[local, view].copy())
            item[pose_key] = torch.from_numpy(poses[local, view].copy())
        return item

def attach_plucker_calibration(cfg, dataset):
    if not (cfg.use_plucker or cfg.plucker_reasoning or getattr(cfg, 'use_camera_tokens', False)
            or getattr(cfg, 'motion_reasoning', False)):
        return dataset
    if not cfg.plucker_calibration_dir:
        raise ValueError('Plucker training requires plucker_calibration_dir')
    if cfg.plucker_convention != 'opengl':
        raise ValueError('the generated MuJoCo calibration sidecars use OpenGL camera axes')
    return ICWMPluckerCalibratedDataset(dataset, cfg.plucker_calibration_dir, getattr(cfg, 'plucker_frame', 'world'))
