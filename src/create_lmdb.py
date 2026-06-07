import argparse
import os
import multiprocessing
from pathlib import Path

import cv2
import lmdb
import numpy as np
from tqdm import tqdm


def read_image_pair(pair):
    """Worker function to read and validate image pairs."""
    gt_path, noisy_path, idx = pair
    gt_img = cv2.imread(str(gt_path), cv2.IMREAD_COLOR)
    noisy_img = cv2.imread(str(noisy_path), cv2.IMREAD_COLOR)

    if gt_img is None or noisy_img is None:
        return None

    # We return the raw data and shape string to minimize overhead
    shape_val = f"{gt_img.shape[0]},{gt_img.shape[1]},{gt_img.shape[2]}"
    return {
        "idx": idx,
        "scene_name": gt_path.parent.name, # Capture the scene folder name
        "gt": np.ascontiguousarray(gt_img).tobytes(),
        "noisy": np.ascontiguousarray(noisy_img).tobytes(),
        "shape": shape_val
    }


def create_lmdb(data_dir, lmdb_path, commit_interval=100, num_workers=None):
    """Converts SIDD PNG pairs into a high-speed LMDB database."""
    # Only run on the main process if using accelerate
    if os.environ.get("RANK", "0") != "0":
        return

    data_dir = Path(data_dir)
    lmdb_path = Path(lmdb_path)

    # Skip if LMDB already exists and is likely complete
    if (lmdb_path / "data.mdb").exists():
        print(f"LMDB already exists at {lmdb_path}. Skipping creation.")
        return

    print(f"Scanning directory: {data_dir}")

    # SIDD structure: Scene_Instance -> (GT_x.PNG, NOISY_x.PNG)
    image_pairs = []
    # Using a more specific glob pattern to speed up scanning
    all_dirs = [d for d in data_dir.iterdir() if d.is_dir()]
    
    for entry in tqdm(all_dirs, desc="Scanning for pairs"):
        gt_files = sorted(list(entry.glob("*GT*.PNG")))
        noisy_files = sorted(list(entry.glob("*NOISY*.PNG")))

        if len(gt_files) != len(noisy_files):
            print(f"Warning: Mismatch in GT ({len(gt_files)}) and NOISY ({len(noisy_files)}) files in {entry}")
            continue

        if len(gt_files) > 0:
            for gt, noisy in zip(gt_files, noisy_files, strict=True):
                # Extra safety check: ensure they belong to the same instance
                gt_prefix = gt.name.replace("_GT.PNG", "").replace("GT", "")
                noisy_prefix = noisy.name.replace("_NOISY.PNG", "").replace("NOISY", "")
                
                if gt_prefix != noisy_prefix:
                     print(f"Warning: Filename mismatch - {gt.name} and {noisy.name}. Skipping.")
                     continue
                
                # We store the global index now to keep keys stable
                image_pairs.append((gt, noisy, len(image_pairs)))

    total_pairs = len(image_pairs)
    print(f"Found {total_pairs} image pairs. Creating LMDB...")

    # Estimate map size (100GB to be safe)
    map_size = 100 * 1024 * 1024 * 1024
    env = lmdb.open(str(lmdb_path), map_size=map_size, writemap=True)

    if num_workers is None:
        num_workers = min(multiprocessing.cpu_count(), 8)

    print(f"Using {num_workers} workers for parallel reading...")
    
    txn = env.begin(write=True)
    processed_count = 0
    
    # Process in chunks to balance memory and speed
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
                # Use scene name in keys to enable scene-based splitting in the dataset
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
                print(f"Processed {min(i + chunk_size, total_pairs)}/{total_pairs} images...")

    txn.commit()
    env.close()
    print(f"LMDB creation complete! Total images stored: {processed_count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="dataset/Data")
    parser.add_argument("--lmdb_dir", type=str, default="dataset/sidd_lmdb")
    parser.add_argument("--workers", type=int, default=None, help="Number of parallel workers")
    args = parser.parse_args()

    lmdb_dir = Path(args.lmdb_dir)
    lmdb_dir.mkdir(parents=True, exist_ok=True)
    create_lmdb(args.data_dir, lmdb_dir, num_workers=args.workers)
