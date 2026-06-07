"""Ensembles multiple SIDD denoising results and generates a Kaggle-compatible CSV submission."""

import scipy.io
import numpy as np
import base64
import csv
import glob
from typing import Optional, List


def array_to_base64string(x: np.ndarray) -> str:
    """Converts a numpy array to a base64 encoded string.

    Args:
        x: The numpy array to convert.

    Returns:
        A base64 encoded string representation of the array.
    """
    return base64.b64encode(x.tobytes()).decode("utf-8")


def load_denoised_array(file_path: str) -> Optional[np.ndarray]:
    """Loads the denoised array from a .mat file, handling historical variable names.

    Args:
        file_path: Path to the .mat file.

    Returns:
        The denoised numpy array if found, otherwise None.
    """
    try:
        data = scipy.io.loadmat(file_path)
        for key in ["Idenoised", "DenoisedBlocksSrgb"]:
            if key in data:
                print(f"  - Loaded '{key}' from {file_path}")
                return data[key]

        for key, val in data.items():
            if isinstance(val, np.ndarray) and val.ndim >= 4:
                print(f"  - Found alternative key '{key}' in {file_path}")
                return val

        raise KeyError(f"No valid denoised array found in {file_path}")
    except Exception as e:
        print(f"  - ERROR loading {file_path}: {e}")
        return None


def main() -> None:
    """Main function to ensemble multiple SIDD denoising results.

    Finds all .mat files in the current directory, averages them in float32 space,
    and saves the result as both a .mat file and a Kaggle-compatible CSV.
    """
    print("=== SIDD Ensemble Podium Push ===")

    exclude_files = [
        "BenchmarkNoisyBlocksSrgb.mat",
        "out.mat",
        "Submit_Ensemble_Final.mat",
    ]
    mat_candidates = glob.glob("*.mat")
    mat_files = [f for f in mat_candidates if f not in exclude_files]

    if not mat_files:
        print("No prediction .mat files found in the current directory.")
        print("Please ensure your SubmitSrgb_version_X.mat files are here.")
        return

    print(f"Found {len(mat_files)} files to ensemble: {', '.join(mat_files)}")

    valid_arrays: List[np.ndarray] = []
    for f in mat_files:
        arr = load_denoised_array(f)
        if arr is not None:
            valid_arrays.append(arr.astype(np.float32))

    if not valid_arrays:
        print("No valid arrays could be loaded. Aborting.")
        return

    print(f"Averaging {len(valid_arrays)} models in float32 space...")
    accumulated = sum(valid_arrays)
    final_float = accumulated / len(valid_arrays)

    final_denoised = np.round(np.clip(final_float, 0, 255)).astype(np.uint8)

    output_base = "Submit_Ensemble_Final"
    scipy.io.savemat(
        f"{output_base}.mat",
        {"Idenoised": final_denoised, "DenoisedBlocksSrgb": final_denoised},
    )
    print(f"Saved ensemble MAT to {output_base}.mat")

    csv_path = f"{output_base}.csv"
    print(f"Generating Kaggle CSV: {csv_path}...")

    with open(csv_path, "w", newline="") as f_out:
        writer = csv.writer(f_out)
        writer.writerow(["ID", "BLOCK"])

        num_scenes, num_blocks = final_denoised.shape[:2]
        total_blocks = num_scenes * num_blocks

        idx = 0
        for s in range(num_scenes):
            for b in range(num_blocks):
                block = np.ascontiguousarray(final_denoised[s, b])
                b64 = array_to_base64string(block)
                writer.writerow([idx, b64])
                idx += 1

    print(f"Ensemble Complete! {idx}/{total_blocks} blocks processed.")
    print("Ready for submission.")


if __name__ == "__main__":
    main()
