import pandas as pd
import numpy as np
import random
from pathlib import Path
from tqdm import tqdm

output_path = "../data/cleaned_low_data_devices_removed_random_noise_devices/"
output_name = "cleaned_low_data_devices_removed_random_noise_devices.csv"
Path(output_path).mkdir(parents=True, exist_ok=True)

original_data_path = "../data/cleaned_low_data_devices_removed/cleaned_low_data_devices_removed-per-packet-hashes"
original_df = pd.read_csv(original_data_path)

num_samples_to_generate = 85950
# num_samples_to_generate = 309418
num_digits_in_hash = 32
label = "other"
random_list = []

for i in tqdm(range(num_samples_to_generate)):
    temp_random_hash = []
    for j in range(num_digits_in_hash):
        temp_random_hash.append(random.randint(0, 255))
    temp_random_hash.append(label)
    random_list.append(temp_random_hash)

# original_np_array = np.array(original_df.values.tolist())
# random_np_array = np.array(random_list)
# output_np_array = np.concatenate((np.array(original_df.values.tolist()), np.array(random_list)), axis=0)
output_list = np.concatenate((np.array(random_list), np.array(original_df.values.tolist())), axis=0).tolist()
# output_list = output_np_array.tolist()

output_df = pd.DataFrame(output_list)
# output_df = output_df.append(original_df, ignore_index=True)
# output_df = pd.concat((original_df, output_df), axis=0, ignore_index=True)
# output_df = output_df.apply(lambda x: pd.Series(x.dropna().values))
# original_df.to_csv(output_path + "test.csv", header=False, index=False)
output_df.to_csv(output_path + output_name, header=False, index=False)