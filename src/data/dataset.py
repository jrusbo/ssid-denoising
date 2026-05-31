import random

import cv2
import lmdb
import numpy as np
import torch
from torch.utils.data import Dataset
from data.augmentations import apply_noise_cutmix, adversarial_frequency_mixup


class SIDDDatasetLMDB(Dataset):
    def __init__(self, lmdb_dir, patch_size=128, split="train", split_ratio=0.9, seed=42):
        super().__init__()
        self.lmdb_dir = lmdb_dir
        self.patch_size = patch_size
        self.split = split
        self.env = None # Will be lazily initialized in __getitem__
        
        # Temporarily open LMDB to get keys
        temp_env = lmdb.open(
            str(self.lmdb_dir), readonly=True, lock=False, readahead=False, meminit=False
        )
        with temp_env.begin() as txn:
            all_keys = sorted([
                key.decode("ascii") for key, _ in txn.cursor() if key.endswith(b"_gt")
            ])
        temp_env.close()

        # Deterministic split: Use a local random instance to avoid side effects
        rng = random.Random(seed)
        rng.shuffle(all_keys)
        split_idx = int(len(all_keys) * split_ratio)
        
        if split == "train":
            self.keys = all_keys[:split_idx]
        else:
            self.keys = all_keys[split_idx:]
            
        self.num_images = len(self.keys)

    def _init_lmdb(self):
        """Initializes the LMDB environment for the current process."""
        self.env = lmdb.open(
            str(self.lmdb_dir), 
            readonly=True, 
            lock=False, 
            readahead=False, 
            meminit=False, 
            max_readers=126
        )

    def __len__(self):
        # For training, we use a large virtual epoch to leverage random crops.
        # For validation, we use the actual number of images for a precise score.
        if self.split == "train":
            return self.num_images * 50
        return self.num_images

    def _augment(self, gt, noisy):
        """Standard geometric self-ensembling (flips and 90/180/270 rotations)."""
        hflip = random.random() < 0.5
        vflip = random.random() < 0.5
        rot90 = random.random() < 0.5

        if hflip:
            gt, noisy = gt[:, ::-1, :], noisy[:, ::-1, :]
        if vflip:
            gt, noisy = gt[::-1, :, :], noisy[::-1, :, :]
        if rot90:
            gt, noisy = gt.transpose(1, 0, 2), noisy.transpose(1, 0, 2)

        return np.ascontiguousarray(gt), np.ascontiguousarray(noisy)

    def _get_crop(self, idx):
        """Gets a decoded and cropped image pair from LMDB for a given index."""
        if self.env is None:
            self._init_lmdb()

        img_idx = idx % self.num_images
        gt_key = self.keys[img_idx]
        noisy_key = gt_key.replace("_gt", "_noisy")

        with self.env.begin() as txn:
            gt_buf = txn.get(gt_key.encode("ascii"))
            noisy_buf = txn.get(noisy_key.encode("ascii"))

            if gt_buf is None or noisy_buf is None:
                raise KeyError(f"Keys {gt_key} or {noisy_key} not found in LMDB")

            gt_img = cv2.imdecode(np.frombuffer(gt_buf, np.uint8), cv2.IMREAD_COLOR)
            noisy_img = cv2.imdecode(np.frombuffer(noisy_buf, np.uint8), cv2.IMREAD_COLOR)

        gt_img = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB)
        noisy_img = cv2.cvtColor(noisy_img, cv2.COLOR_BGR2RGB)

        H, W, _ = gt_img.shape

        # Prevent out-of-bounds by padding if the image is smaller than the requested patch_size
        pad_h = max(0, self.patch_size - H)
        pad_w = max(0, self.patch_size - W)
        if pad_h > 0 or pad_w > 0:
            gt_img = np.pad(gt_img, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
            noisy_img = np.pad(noisy_img, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
            H, W, _ = gt_img.shape

        if self.split == "train":
            rnd_h = random.randint(0, H - self.patch_size)
            rnd_w = random.randint(0, W - self.patch_size)
            gt_crop = gt_img[rnd_h : rnd_h + self.patch_size, rnd_w : rnd_w + self.patch_size, :]
            noisy_crop = noisy_img[rnd_h : rnd_h + self.patch_size, rnd_w : rnd_w + self.patch_size, :]
            gt_crop, noisy_crop = self._augment(gt_crop, noisy_crop)
        else:
            start_h = (H - self.patch_size) // 2
            start_w = (W - self.patch_size) // 2
            gt_crop = gt_img[start_h : start_h + self.patch_size, start_w : start_w + self.patch_size, :]
            noisy_crop = noisy_img[start_h : start_h + self.patch_size, start_w : start_w + self.patch_size, :]

        gt_tensor = torch.from_numpy(gt_crop).float().permute(2, 0, 1) / 255.0
        noisy_tensor = torch.from_numpy(noisy_crop).float().permute(2, 0, 1) / 255.0

        return noisy_tensor, gt_tensor

    def __getitem__(self, idx):
        # Lazy initialization
        return self._get_crop(idx)
