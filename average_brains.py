import os
import random
import numpy as np

# --- Settings ---
folder_path = '.'  # The current directory
num_samples = 150  # Number of files to sample (adjust as needed)
target_suffix = 'cat12vbm_desc-gm_T1w.npy' # The specific files we want
# ----------------

# 1. Find all matching .npy files
all_files = [f for f in os.listdir(folder_path) if f.endswith(target_suffix)]

# Check if we have enough files
if len(all_files) < num_samples:
    print(f"Found only {len(all_files)} files, but requested {num_samples}.")
    num_samples = len(all_files)

# 2. Randomly sample the files
sampled_files = random.sample(all_files, num_samples)
print(f"Found {len(all_files)} files. Averaging {num_samples} of them...")

# 3. Calculate the running sum to save RAM
sum_array = None

for count, file in enumerate(sampled_files, 1):
    file_path = os.path.join(folder_path, file)
    
    # Load the current brain array
    current_data = np.load(file_path)
    
    # On the first pass, create an empty array of the exact same shape to hold our sums.
    # We use float64 to ensure the numbers don't overflow while adding them all up.
    if sum_array is None:
        sum_array = np.zeros(current_data.shape, dtype=np.float64)
    
    # Add current brain to the running total
    sum_array += current_data
    
    # Print progress every 25 files
    if count % 25 == 0:
        print(f"Processed {count}/{num_samples} files...")

# 4. Divide by the total number of samples to get the mean
average_array = sum_array / num_samples

# 5. Save the final averaged array
output_name = 'group_average_gm_T1w.npy'
np.save(output_name, average_array)
print(f"Success! Saved the average to: {output_name}")