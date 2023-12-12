import os
import shutil

# path_to_hashes = "/home/nthom/Downloads/similar_devices_Hashes/"
# path_to_copy = "/home/nthom/Documents/SmartRecon/Fingerprinting in Noisy Network Environments/data/simhashes/"
path_to_hashes = "/home/nthom/Downloads/FlexHash_noise/"
path_to_copy = "/home/nthom/Documents/SmartRecon/Fingerprinting in Noisy Network Environments/data/FlexHash_noise/"

csv_file_list = sorted(os.listdir(path_to_hashes))

for filename in csv_file_list:
    if filename[:3] == "cam":
        device = "cam"
    elif filename[:4] == "plug":
        device = "plug"
    elif filename[:5] == "light":
        device = "light"
    elif filename[:13] == "network_noise":
        device = "network_noise"
    elif filename[:9] == "iot_noise":
        device = "iot_noise"

    filename_split = filename[:-4].split("_")
    print(filename_split)

    accum = filename_split[-3]
    window = filename_split[-2]
    combo = filename_split[-1]

    os.makedirs(
        f"{path_to_copy}{device}/accum_{accum}/window_{window}/combo_{combo}/",
        exist_ok=True,
    )
    shutil.copy(
        path_to_hashes + filename,
        f"{path_to_copy}{device}/accum_{accum}/window_{window}/combo_{combo}/"
        + filename,
    )