import argparse
import os
from pathlib import Path

import cv2
import lmdb
from tqdm import tqdm


def create_lmdb(data_dir, lmdb_path, commit_interval=100):
    """Converts SIDD PNG pairs into a high-speed LMDB database."""
    # Only run on the main process if using accelerate
    if os.environ.get("RANK", "0") != "0":
        return

    data_dir = Path(data_dir)
    lmdb_path = Path(lmdb_path)

    print(f"Scanning directory: {data_dir}")

    # SIDD structure: Scene_Instance -> (GT_x.PNG, NOISY_x.PNG)
    # We will collect all paired paths first
    image_pairs = []
    for entry in data_dir.rglob("*"):
        if entry.is_dir():
            gt_files = sorted(list(entry.glob("*GT*.PNG")))
            noisy_files = sorted(list(entry.glob("*NOISY*.PNG")))

            if len(gt_files) != len(noisy_files):
                print(f"Warning: Mismatch in GT ({len(gt_files)}) and NOISY ({len(noisy_files)}) files in {entry}")
                continue

            if len(gt_files) > 0:
                for gt, noisy in zip(gt_files, noisy_files, strict=True):
                    # Extra safety check: ensure they belong to the same instance if possible
                    gt_prefix = gt.name.replace("_GT.PNG", "").replace("GT", "")
                    noisy_prefix = noisy.name.replace("_NOISY.PNG", "").replace("NOISY", "")
                    
                    if gt_prefix != noisy_prefix:
                         print(f"Warning: Filename mismatch - {gt.name} and {noisy.name}. Skipping.")
                         continue
                         
                    image_pairs.append((gt, noisy))

    print(f"Found {len(image_pairs)} image pairs. Creating LMDB...")

    # Estimate map size (100GB to be safe, LMDB dynamically allocates what it needs)
    map_size = 100 * 1024 * 1024 * 1024
    env = lmdb.open(str(lmdb_path), map_size=map_size, writemap=True)

    txn = env.begin(write=True)
    for idx, (gt_path, noisy_path) in enumerate(tqdm(image_pairs)):
        # Read images using OpenCV (BGR format by default)
        gt_img = cv2.imread(str(gt_path), cv2.IMREAD_COLOR)
        noisy_img = cv2.imread(str(noisy_path), cv2.IMREAD_COLOR)

        if gt_img is None or noisy_img is None:
            print(f"Warning: Could not read {gt_path} or {noisy_path}. Skipping.")
            continue

        # Encode to save space and ensure contiguous memory
        # OpenCV imencode expects BGR images.
        _, gt_encoded = cv2.imencode(".png", gt_img)
        _, noisy_encoded = cv2.imencode(".png", noisy_img)

        # Store in LMDB with strict key naming conventions
        gt_key = f"{idx:05d}_gt".encode("ascii")
        noisy_key = f"{idx:05d}_noisy".encode("ascii")

        txn.put(gt_key, gt_encoded.tobytes())
        txn.put(noisy_key, noisy_encoded.tobytes())

        if (idx + 1) % commit_interval == 0:
            txn.commit()
            txn = env.begin(write=True)

    txn.commit()
    env.close()
    print(f"LMDB creation complete! Total images: {len(image_pairs)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="dataset/Data")
    parser.add_argument("--lmdb_dir", type=str, default="dataset/sidd_lmdb")
    args = parser.parse_args()

    lmdb_dir = Path(args.lmdb_dir)
    lmdb_dir.mkdir(parents=True, exist_ok=True)
    create_lmdb(args.data_dir, lmdb_dir)
