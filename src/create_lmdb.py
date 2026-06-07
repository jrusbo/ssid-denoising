"""Utility script to convert SIDD PNG image pairs into a high-performance LMDB database."""

import argparse
import os
import multiprocessing
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import cv2
import lmdb
import numpy as np
from tqdm import tqdm


def read_image_pair(
    pair: Tuple[Path, Path, int],
) -> Optional[Dict[str, Union[int, str, bytes]]]:
    """Worker function to read and validate image pairs.

    Args:
        pair: A tuple containing (gt_path, noisy_path, index).

    Returns:
        A dictionary with image data and metadata, or None if reading fails.
    """
    gt_path, noisy_path, idx = pair
    gt_img = cv2.imread(str(gt_path), cv2.IMREAD_COLOR)
    noisy_img = cv2.imread(str(noisy_path), cv2.IMREAD_COLOR)

    if gt_img is None or noisy_img is None:
        return None

    shape_val = f"{gt_img.shape[0]},{gt_img.shape[1]},{gt_img.shape[2]}"
    return {
        "idx": idx,
        "scene_name": gt_path.parent.name,
        "gt": np.ascontiguousarray(gt_img).tobytes(),
        "noisy": np.ascontiguousarray(noisy_img).tobytes(),
        "shape": shape_val,
    }


def create_lmdb(
    data_dir: Union[str, Path],
    lmdb_path: Union[str, Path],
    commit_interval: int = 100,
    num_workers: Optional[int] = None,
) -> None:
    """Converts SIDD PNG pairs into a high-speed LMDB database.

    Args:
        data_dir: Directory containing SIDD scene folders.
        lmdb_path: Path where the LMDB database will be created.
        commit_interval: Number of images to process before committing to LMDB.
        num_workers: Number of parallel workers for image reading.
    """
    if os.environ.get("RANK", "0") != "0":
        return

    data_dir = Path(data_dir)
    lmdb_path = Path(lmdb_path)

    if (lmdb_path / "data.mdb").exists():
        print(f"LMDB already exists at {lmdb_path}. Skipping creation.")
        return

    print(f"Scanning directory: {data_dir}")

    image_pairs = []
    all_dirs = [d for d in data_dir.iterdir() if d.is_dir()]

    for entry in tqdm(all_dirs, desc="Scanning for pairs"):
        gt_files = sorted(list(entry.glob("*GT*.PNG")))
        noisy_files = sorted(list(entry.glob("*NOISY*.PNG")))

        if len(gt_files) != len(noisy_files):
            print(
                f"Warning: Mismatch in GT ({len(gt_files)}) and NOISY ({len(noisy_files)}) files in {entry}"
            )
            continue

        if len(gt_files) > 0:
            for gt, noisy in zip(gt_files, noisy_files, strict=True):
                gt_prefix = gt.name.replace("_GT.PNG", "").replace("GT", "")
                noisy_prefix = noisy.name.replace("_NOISY.PNG", "").replace("NOISY", "")

                if gt_prefix != noisy_prefix:
                    print(
                        f"Warning: Filename mismatch - {gt.name} and {noisy.name}. Skipping."
                    )
                    continue

                image_pairs.append((gt, noisy, len(image_pairs)))

    total_pairs = len(image_pairs)
    print(f"Found {total_pairs} image pairs. Creating LMDB...")

    # Estimate map size (100GB)
    map_size = 100 * 1024 * 1024 * 1024
    env = lmdb.open(str(lmdb_path), map_size=map_size, writemap=True)

    if num_workers is None:
        num_workers = min(multiprocessing.cpu_count(), 8)

    print(f"Using {num_workers} workers for parallel reading...")

    txn = env.begin(write=True)
    processed_count = 0

    chunk_size = num_workers * 4
    with multiprocessing.Pool(num_workers) as pool:
        for i in range(0, total_pairs, chunk_size):
            chunk = image_pairs[i : i + chunk_size]
            results = pool.map(read_image_pair, chunk)

            for res in results:
                if res is None:
                    continue

                idx = res["idx"]
                scene_name = res["scene_name"]
                gt_key = f"{scene_name}_{idx:06d}_gt".encode("ascii")
                noisy_key = f"{scene_name}_{idx:06d}_noisy".encode("ascii")
                shape_key = f"{scene_name}_{idx:06d}_shape".encode("ascii")

                txn.put(gt_key, res["gt"])
                txn.put(noisy_key, res["noisy"])
                txn.put(shape_key, res["shape"].encode("ascii"))

                processed_count += 1

                if processed_count % commit_interval == 0:
                    txn.commit()
                    txn = env.begin(write=True)

            if (i // chunk_size) % 5 == 0:
                print(
                    f"Processed {min(i + chunk_size, total_pairs)}/{total_pairs} images..."
                )

    txn.commit()
    env.close()
    print(f"LMDB creation complete! Total images stored: {processed_count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="dataset/Data")
    parser.add_argument("--lmdb_dir", type=str, default="dataset/sidd_lmdb")
    parser.add_argument(
        "--workers", type=int, default=None, help="Number of parallel workers"
    )
    args = parser.parse_args()

    lmdb_dir = Path(args.lmdb_dir)
    lmdb_dir.mkdir(parents=True, exist_ok=True)
    create_lmdb(args.data_dir, lmdb_dir, num_workers=args.workers)
