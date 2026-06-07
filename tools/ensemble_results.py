
import scipy.io
import numpy as np
import base64
import csv
import os
import glob

def array_to_base64string(x):
    """Converts a numpy array to a base64 encoded string."""
    return base64.b64encode(x.tobytes()).decode("utf-8")

def load_denoised_array(file_path):
    """Loads the denoised array, handling different variable names from project history."""
    try:
        data = scipy.io.loadmat(file_path)
        # Check for the two common keys used in your project history
        for key in ['Idenoised', 'DenoisedBlocksSrgb']:
            if key in data:
                print(f"  - Loaded '{key}' from {file_path}")
                return data[key]
        
        # Fallback: find any key that looks like an image array (4D or 5D)
        for key, val in data.items():
            if isinstance(val, np.ndarray) and val.ndim >= 4:
                print(f"  - Found alternative key '{key}' in {file_path}")
                return val
                
        raise KeyError(f"No valid denoised array found in {file_path}")
    except Exception as e:
        print(f"  - ERROR loading {file_path}: {e}")
        return None

def main():
    print("=== SIDD Ensemble Podium Push ===")
    
    # 1. Automatically find all .mat files in the current folder
    # We exclude the benchmark file and any known output files
    exclude_files = ['BenchmarkNoisyBlocksSrgb.mat', 'out.mat', 'Submit_Ensemble_Final.mat']
    mat_candidates = glob.glob("*.mat")
    mat_files = [f for f in mat_candidates if f not in exclude_files]
    
    if not mat_files:
        print("No prediction .mat files found in the current directory.")
        print("Please ensure your SubmitSrgb_version_X.mat files are here.")
        return

    print(f"Found {len(mat_files)} files to ensemble: {', '.join(mat_files)}")

    # 2. Load and accumulate
    valid_arrays = []
    for f in mat_files:
        arr = load_denoised_array(f)
        if arr is not None:
            # Convert to float32 immediately to prevent overflow during addition
            valid_arrays.append(arr.astype(np.float32))

    if not valid_arrays:
        print("No valid arrays could be loaded. Aborting.")
        return

    # 3. High-Precision Averaging
    print(f"Averaging {len(valid_arrays)} models in float32 space...")
    accumulated = sum(valid_arrays)
    final_float = accumulated / len(valid_arrays)
    
    # Final rounding to uint8 [0, 255]
    final_denoised = np.round(np.clip(final_float, 0, 255)).astype(np.uint8)

    # 4. Save Final MAT (Double-keyed for compatibility)
    output_base = "Submit_Ensemble_Final"
    scipy.io.savemat(f"{output_base}.mat", {
        "Idenoised": final_denoised,
        "DenoisedBlocksSrgb": final_denoised
    })
    print(f"Saved ensemble MAT to {output_base}.mat")

    # 5. Generate Kaggle CSV
    csv_path = f"{output_base}.csv"
    print(f"Generating Kaggle CSV: {csv_path}...")
    
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ID", "BLOCK"])
        
        # SIDD Benchmark format: (scenes, blocks, H, W, C)
        # Standard: 40 scenes, 32 blocks each = 1280 blocks total
        num_scenes, num_blocks = final_denoised.shape[:2]
        total_blocks = num_scenes * num_blocks
        
        idx = 0
        for s in range(num_scenes):
            for b in range(num_blocks):
                # Ensure the block is contiguous for proper base64 encoding
                block = np.ascontiguousarray(final_denoised[s, b])
                b64 = array_to_base64string(block)
                writer.writerow([idx, b64])
                idx += 1
                
    print(f"Ensemble Complete! {idx}/{total_blocks} blocks processed.")
    print("Ready for submission.")

if __name__ == "__main__":
    main()
