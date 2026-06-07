import random
from collections import defaultdict
from pathlib import Path
from typing import Optional, Tuple, Union

import lmdb
import numpy as np
import torch
from torch.utils.data import Dataset


class SIDDDatasetLMDB(Dataset):
    """Dataset class for loading SIDD data from an LMDB database.

    Attributes:
        lmdb_dir: Path to the LMDB database directory.
        patch_size: Size of the image patches to extract.
        split: Dataset split ('train' or 'val').
        env: LMDB environment instance.
        keys: List of keys for accessing data in LMDB.
        num_images: Total number of unique images in the dataset.
    """

    def __init__(
        self,
        lmdb_dir: Union[str, Path],
        patch_size: int = 128,
        split: str = "train",
        split_ratio: float = 0.9,
        seed: int = 42,
    ) -> None:
        """Initializes the dataset.

        Args:
            lmdb_dir: Path to the LMDB directory.
            patch_size: Patch size for cropping.
            split: 'train' or 'val'.
            split_ratio: Ratio of scenes to use for training.
            seed: Random seed for deterministic split.
        """
        super().__init__()
        self.lmdb_dir = Path(lmdb_dir)
        self.patch_size = patch_size
        self.split = split
        self.env: Optional[lmdb.Environment] = None

        # Temporarily open LMDB to get keys
        temp_env = lmdb.open(
            str(self.lmdb_dir),
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        with temp_env.begin() as txn:
            all_keys = sorted(
                [key.decode("ascii") for key, _ in txn.cursor() if key.endswith(b"_gt")]
            )
        temp_env.close()

        # Group keys by scene ID for deterministic split
        scene_to_keys = defaultdict(list)
        for key in all_keys:
            scene_id = key.split("_")[0]
            scene_to_keys[scene_id].append(key)

        unique_scenes = sorted(list(scene_to_keys.keys()))

        rng = random.Random(seed)
        rng.shuffle(unique_scenes)

        num_scenes = len(unique_scenes)
        if num_scenes == 0:
            raise ValueError(
                f"No valid scenes found in LMDB directory: {self.lmdb_dir}"
            )

        split_idx = max(1, min(num_scenes - 1, int(num_scenes * split_ratio)))

        if split == "train":
            selected_scenes = unique_scenes[:split_idx]
        else:
            selected_scenes = unique_scenes[split_idx:]

        self.keys = []
        for scene in selected_scenes:
            self.keys.extend(scene_to_keys[scene])

        self.num_images = len(self.keys)
        print(
            f"Dataset split ({split}): {len(selected_scenes)} scenes, {self.num_images} images."
        )

    def _init_lmdb(self) -> None:
        """Initializes the LMDB environment for the current process."""
        self.env = lmdb.open(
            str(self.lmdb_dir),
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=126,
        )

    def __len__(self) -> int:
        """Returns the length of the dataset."""
        if self.split == "train":
            # For training, we use a large virtual epoch to leverage random crops.
            # For validation, we use the actual number of images for a precise score.
            return self.num_images * 50
        return self.num_images

    def _augment(
        self, gt: np.ndarray, noisy: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Standard geometric augmentations (flips and rotations)."""
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

    def _get_crop(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Gets a decoded and cropped image pair from LMDB.

        Args:
            idx: Index of the image to retrieve.

        Returns:
            A tuple of (noisy_tensor, gt_tensor).
        """
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
                raise KeyError(f"Data for {gt_key} not found in LMDB")

            H, W, C = map(int, bytes(shape_buf).decode("ascii").split(","))
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
            def get_fast_padded_crop(
                buf: Union[bytes, memoryview],
                h_start: int,
                w_start: int,
                p_size: int,
                curr_h: int,
                curr_w: int,
            ) -> np.ndarray:
                h_end = min(h_start + p_size, curr_h)
                w_end = min(w_start + p_size, curr_w)

                start_idx = h_start * curr_w * C
                end_idx = h_end * curr_w * C

                mv = memoryview(buf)
                row_block = np.frombuffer(mv[start_idx:end_idx], dtype=np.uint8).copy()
                row_block = row_block.reshape(h_end - h_start, curr_w, C)

                crop = row_block[:, w_start:w_end, :].copy()

                pad_h = p_size - crop.shape[0]
                pad_w = p_size - crop.shape[1]
                if pad_h > 0 or pad_w > 0:
                    crop = np.pad(
                        crop, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect"
                    )
                return crop

            gt_crop = get_fast_padded_crop(gt_buf, rnd_h, rnd_w, self.patch_size, H, W)
            noisy_crop = get_fast_padded_crop(
                noisy_buf, rnd_h, rnd_w, self.patch_size, H, W
            )

        # Convert BGR (LMDB default) to RGB
        gt_crop = np.ascontiguousarray(gt_crop[:, :, ::-1])
        noisy_crop = np.ascontiguousarray(noisy_crop[:, :, ::-1])

        if self.split == "train":
            gt_crop, noisy_crop = self._augment(gt_crop, noisy_crop)

        inv_255 = 1.0 / 255.0
        gt_tensor = (
            torch.from_numpy(gt_crop)
            .permute(2, 0, 1)
            .to(dtype=torch.float32)
            .mul_(inv_255)
        )
        noisy_tensor = (
            torch.from_numpy(noisy_crop)
            .permute(2, 0, 1)
            .to(dtype=torch.float32)
            .mul_(inv_255)
        )

        return noisy_tensor, gt_tensor

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns the item at the specified index."""
        return self._get_crop(idx)
