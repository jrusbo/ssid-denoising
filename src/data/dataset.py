import random

import lmdb
import numpy as np
import torch
from torch.utils.data import Dataset



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
        shape_key = gt_key.replace("_gt", "_shape")

        with self.env.begin(buffers=True) as txn:
            gt_buf = txn.get(gt_key.encode("ascii"))
            noisy_buf = txn.get(noisy_key.encode("ascii"))
            shape_buf = txn.get(shape_key.encode("ascii"))

            if gt_buf is None or noisy_buf is None or shape_buf is None:
                raise KeyError(f"Data for {gt_key} not found in LMDB (Missing bytes or shape)")

            # Parse shape from stored metadata "H,W,C"
            H, W, C = map(int, bytes(shape_buf).decode("ascii").split(","))

            # Determine crop coordinates first
            effective_H, effective_W = max(H, self.patch_size), max(W, self.patch_size)

            if self.split == "train":
                rnd_h = random.randint(0, effective_H - self.patch_size)
                rnd_w = random.randint(0, effective_W - self.patch_size)
            else:
                rnd_h = (effective_H - self.patch_size) // 2
                rnd_w = (effective_W - self.patch_size) // 2

            # Fast memoryview crop to avoid doing highly scattered 2D reads from a 36MB mmap file.
            # Doing 2D slices natively across 8 Python workers on Windows will lock the OS memory manager and thrash page faults.
            # Instead, we pull one contiguous 1D block containing the required rows sequentially, then slice the width in RAM.
            def get_fast_padded_crop(buf, h_start, w_start, p_size, curr_h, curr_w):
                h_end = min(h_start + p_size, curr_h)
                w_end = min(w_start + p_size, curr_w)
                
                # 1D linear slice from LMDB memory view
                start_idx = h_start * curr_w * C
                end_idx = h_end * curr_w * C

                # Sequentially copy the row block into RAM (approx ~1-2 MB instead of 36MB), breaking the mmap pointer.
                mv = memoryview(buf)
                row_block = np.frombuffer(mv[start_idx:end_idx], dtype=np.uint8).copy()
                row_block = row_block.reshape(h_end - h_start, curr_w, C)

                # Extract width crop locally
                crop = row_block[:, w_start:w_end, :].copy()

                # Pad if the crop is smaller than p_size
                pad_h = p_size - crop.shape[0]
                pad_w = p_size - crop.shape[1]
                if pad_h > 0 or pad_w > 0:
                    crop = np.pad(crop, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
                return crop

            gt_crop = get_fast_padded_crop(gt_buf, rnd_h, rnd_w, self.patch_size, H, W)
            noisy_crop = get_fast_padded_crop(noisy_buf, rnd_h, rnd_w, self.patch_size, H, W)

        # Convert the SMALL CROP from BGR (OpenCV default during creation) to RGB
        gt_crop = np.ascontiguousarray(gt_crop[:, :, ::-1])
        noisy_crop = np.ascontiguousarray(noisy_crop[:, :, ::-1])

        if self.split == "train":
            gt_crop, noisy_crop = self._augment(gt_crop, noisy_crop)

        gt_tensor = torch.from_numpy(gt_crop).float().permute(2, 0, 1) / 255.0
        noisy_tensor = torch.from_numpy(noisy_crop).float().permute(2, 0, 1) / 255.0

        return noisy_tensor, gt_tensor

    def __getitem__(self, idx):
        # Lazy initialization
        return self._get_crop(idx)
