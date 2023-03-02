import os
import random

import pandas as pd
from pandas import read_csv
import numpy as np
from tqdm import tqdm


def remove_class(class_name, dataset):
    dataset = dataset[dataset["class"] != class_name]
    return dataset


def noise_generator(original_df, names, one_hundred_times=False):
    num_samples_to_generate = len(original_df.index)
    if one_hundred_times == True:
        num_samples_to_generate *= 100
    num_digits_in_hash = 32
    label = "other"
    random_list = []

    for i in tqdm(range(num_samples_to_generate), desc="Generating random noise"):
        temp_random_hash = []
        for j in range(num_digits_in_hash):
            temp_random_hash.append(random.randint(0, 255))
        temp_random_hash.append(label)
        random_list.append(temp_random_hash)

    output_list = np.concatenate(
        (np.array(random_list), np.array(original_df.values.tolist())), axis=0
    ).tolist()
    output_df = pd.DataFrame(output_list, columns=names)
    return output_df


def combine_csv(csv_list, names):
    final_df = pd.DataFrame(columns=names)
    temp_df_list = []
    for csv in tqdm(csv_list):
        # temp_df = pd.read_csv(csv, names=names, skiprows=1)
        temp_df_list.append(pd.read_csv(csv, names=names, skiprows=1))
        # final_df = pd.concat([final_df, temp_df])
    final_df = pd.concat(temp_df_list)

    return final_df

def combine_csv_category(csv_list, names):
    final_df = pd.DataFrame(columns=names)
    temp_df_list = []
    for csv in tqdm(csv_list):
        temp_df = pd.read_csv(csv, names=names, skiprows=1)
        if temp_df["class"][0][:4] == "plug":
            temp_df["class"] = "plug"
        elif temp_df["class"][0][:5] == "light":
            temp_df["class"] = "light"
        else:
            temp_df["class"] = "cam"
        temp_df_list.append(temp_df)
        # final_df = pd.concat([final_df, temp_df])
    final_df = pd.concat(temp_df_list)

    return final_df


path_to_iot_noise_cleaned = "/home/nthom/Documents/SmartRecon/Fingerprinting in Noisy Network Environments/data/noise/iot_noise/iot_noise_hashes_cleaned.csv"
path_to_iot_noise_uncleaned = "/home/nthom/Documents/SmartRecon/Fingerprinting in Noisy Network Environments/data/noise/iot_noise/iot_noise_hashes_uncleaned.csv"

path_to_network_noise_cleaned = "/home/nthom/Documents/SmartRecon/Fingerprinting in Noisy Network Environments/data/noise/network_noise/network_noise_hashes_cleaned.csv"
path_to_network_noise_uncleaned = "/home/nthom/Documents/SmartRecon/Fingerprinting in Noisy Network Environments/data/noise/network_noise/network_noise_hashes_uncleaned.csv"

path_to_per_packet_cleaned_devices = "/home/nthom/Documents/SmartRecon/Fingerprinting in Noisy Network Environments/data/per_packet_hashes/cleaned/per-packet-hashes-cleaned.csv"
path_to_per_packet_uncleaned_devices = "/home/nthom/Documents/SmartRecon/Fingerprinting in Noisy Network Environments/data/per_packet_hashes/uncleaned/per-packet-hashes-uncleaned.csv"

path_to_per_packet_cleaned_categories = "/home/nthom/Documents/SmartRecon/Fingerprinting in Noisy Network Environments/data/per_packet_hashes/cleaned/cleaned-categories.csv"
path_to_per_packet_uncleaned_categories = "/home/nthom/Documents/SmartRecon/Fingerprinting in Noisy Network Environments/data/per_packet_hashes/uncleaned/uncleaned-categories.csv"

path_to_same_plug_cleaned_interaction = "/home/nthom/Documents/SmartRecon/Fingerprinting in Noisy Network Environments/data/same_device/same_plug/same_plug_cleaned_interaction/"
path_to_same_plug_cleaned_no_interaction = "/home/nthom/Documents/SmartRecon/Fingerprinting in Noisy Network Environments/data/same_device/same_plug/same_plug_cleaned_no_interaction/"
path_to_same_plug_uncleaned_interaction = "/home/nthom/Documents/SmartRecon/Fingerprinting in Noisy Network Environments/data/same_device/same_plug/same_plug_uncleaned_interaction/"
path_to_same_plug_uncleaned_no_interaction = "/home/nthom/Documents/SmartRecon/Fingerprinting in Noisy Network Environments/data/same_device/same_plug/same_plug_uncleaned_no_interaction/"

path_to_same_bulb_cleaned_interaction = "/home/nthom/Documents/SmartRecon/Fingerprinting in Noisy Network Environments/data/same_device/same_lightbulb/same_lightbulb_cleaned_interaction/"
path_to_same_bulb_cleaned_no_interaction = "/home/nthom/Documents/SmartRecon/Fingerprinting in Noisy Network Environments/data/same_device/same_lightbulb/same_lightbulb_cleaned_no_interaction/"
path_to_same_bulb_uncleaned_interaction = "/home/nthom/Documents/SmartRecon/Fingerprinting in Noisy Network Environments/data/same_device/same_lightbulb/same_lightbulb_uncleaned_interaction/"
path_to_same_bulb_uncleaned_no_interaction = "/home/nthom/Documents/SmartRecon/Fingerprinting in Noisy Network Environments/data/same_device/same_lightbulb/same_lightbulb_uncleaned_no_interaction/"

path_to_same_cam_cleaned_interaction = "/home/nthom/Documents/SmartRecon/Fingerprinting in Noisy Network Environments/data/same_device/same_cam/same_cam_cleaned_interaction/"
path_to_same_cam_cleaned_no_interaction = "/home/nthom/Documents/SmartRecon/Fingerprinting in Noisy Network Environments/data/same_device/same_cam/same_cam_cleaned_no_interaction/"
path_to_same_cam_uncleaned_interaction = "/home/nthom/Documents/SmartRecon/Fingerprinting in Noisy Network Environments/data/same_device/same_cam/same_cam_uncleaned_interaction/"
path_to_same_cam_uncleaned_no_interaction = "/home/nthom/Documents/SmartRecon/Fingerprinting in Noisy Network Environments/data/same_device/same_cam/same_cam_uncleaned_no_interaction/"


def get_dataset():
    path_to_simhash = "/home/nthom/Documents/SmartRecon/Fingerprinting in Noisy Network Environments/data/simhashes/"

    names = [
        "dim1",
        "dim2",
        "dim3",
        "dim4",
        "dim5",
        "dim6",
        "dim7",
        "dim8",
        "dim9",
        "dim10",
        "dim11",
        "dim12",
        "dim13",
        "dim14",
        "dim15",
        "dim16",
        "dim17",
        "dim18",
        "dim19",
        "dim20",
        "dim21",
        "dim22",
        "dim23",
        "dim24",
        "dim25",
        "dim26",
        "dim27",
        "dim28",
        "dim29",
        "dim30",
        "dim31",
        "dim32",
        "class",
    ]

    experiment_type = int(
        input(
            "Select one of the following: \n1. Nilsimsa Per-Packet Devices \n"
            "2. Nilsimsa Per-Packet Categories \n3. Nilsimsa Identical Devices \n"
            "4. FlexHash Identical Devices \n5. 100x Noise\n6. PvLvC\n"
        )
    )
    if not experiment_type in [1, 2, 3, 4, 5]:
        raise ValueError(
            "'experiment_type' selection must be one of the following values: 1, 2, 3, 4, 5, or 6."
        )

    c_uc = int(input("Select one of the following: \n1. Cleaned \n2. Uncleaned\n"))
    if not c_uc in [1, 2]:
        raise ValueError(
            "'c_uc' selection must be one of the following values: 1 or 2,"
        )

    if experiment_type == 1 and c_uc == 1:
        noise = int(
            input(
                "Select one of the following: \n1. Random \n2. IoT Cleaned \n3. IoT Uncleaned \n"
                "4. Network Cleaned \n5. Network Uncleaned \n6. None\n"
            )
        )
        if not noise in [1, 2, 3, 4, 5, 6]:
            raise ValueError(
                "'noise' selection must be one of the following values: 1, 2, 3, 4, 5 or 6."
            )

        if noise != 6:
            if noise == 1:
                dataset = read_csv(path_to_per_packet_cleaned_devices, names=names)
                dataset = noise_generator(dataset, names)
                name = "cleaned_devices-random"
            elif noise == 2:
                csv_list = [
                    path_to_per_packet_cleaned_devices,
                    path_to_iot_noise_cleaned,
                ]
                dataset = combine_csv(csv_list, names)
                name = "cleaned_devices-cleaned_iot"
            elif noise == 3:
                csv_list = [
                    path_to_per_packet_cleaned_devices,
                    path_to_iot_noise_uncleaned,
                ]
                dataset = combine_csv(csv_list, names)
                name = "cleaned_devices-uncleaned_iot"
            elif noise == 4:
                csv_list = [
                    path_to_per_packet_cleaned_devices,
                    path_to_network_noise_cleaned,
                ]
                dataset = combine_csv(csv_list, names)
                name = "cleaned_devices-cleaned_network"
            else:
                csv_list = [
                    path_to_per_packet_cleaned_devices,
                    path_to_network_noise_uncleaned,
                ]
                dataset = combine_csv(csv_list, names)
                name = "cleaned_devices-uncleaned_network"
        else:
            dataset = read_csv(path_to_per_packet_cleaned_devices, names=names)
            name = "cleaned_devices"
    elif experiment_type == 1 and c_uc == 2:
        noise = int(
            input(
                "Select one of the following: \n1. Random \n2. IoT Cleaned \n3. IoT Uncleaned \n"
                "4. Network Cleaned \n5. Network Uncleaned \n6. None\n"
            )
        )
        if not noise in [1, 2, 3, 4, 5, 6]:
            raise ValueError(
                "'noise' selection must be one of the following values: 1, 2, 3, 4, 5 or 6."
            )

        if noise != 6:
            if noise == 1:
                dataset = read_csv(path_to_per_packet_uncleaned_devices, names=names)
                dataset = noise_generator(dataset, names)
                name = "uncleaned_devices-random"
            elif noise == 2:
                csv_list = [
                    path_to_per_packet_uncleaned_devices,
                    path_to_iot_noise_cleaned,
                ]
                dataset = combine_csv(csv_list, names)
                name = "uncleaned_devices-cleaned_iot"
            elif noise == 3:
                csv_list = [
                    path_to_per_packet_uncleaned_devices,
                    path_to_iot_noise_uncleaned,
                ]
                dataset = combine_csv(csv_list, names)
                name = "uncleaned_devices-uncleaned_iot"
            elif noise == 4:
                csv_list = [
                    path_to_per_packet_uncleaned_devices,
                    path_to_network_noise_cleaned,
                ]
                dataset = combine_csv(csv_list, names)
                name = "uncleaned_devices-cleaned_network"
            else:
                csv_list = [
                    path_to_per_packet_uncleaned_devices,
                    path_to_network_noise_uncleaned,
                ]
                dataset = combine_csv(csv_list, names)
                name = "uncleaned_devices-uncleaned-network"
        else:
            dataset = read_csv(path_to_per_packet_uncleaned_devices, names=names)
            name = "uncleaned_devices"
    elif experiment_type == 2 and c_uc == 1:
        noise = int(
            input(
                "Select one of the following: \n1. Random \n2. IoT Cleaned \n3. IoT Uncleaned \n"
                "4. Network Cleaned \n5. Network Uncleaned \n6. None\n"
            )
        )
        if not noise in [1, 2, 3, 4, 5, 6]:
            raise ValueError(
                "'noise' selection must be one of the following values: 1, 2, 3, 4, 5 or 6."
            )

        if noise != 6:
            if noise == 1:
                dataset = read_csv(path_to_per_packet_cleaned_categories, names=names)
                dataset = noise_generator(dataset, names)
                name = "cleaned_categories-random"
            elif noise == 2:
                csv_list = [
                    path_to_per_packet_cleaned_categories,
                    path_to_iot_noise_cleaned,
                ]
                dataset = combine_csv(csv_list, names)
                name = "cleaned_categories-cleaned_iot"
            elif noise == 3:
                csv_list = [
                    path_to_per_packet_cleaned_categories,
                    path_to_iot_noise_uncleaned,
                ]
                dataset = combine_csv(csv_list, names)
                name = "cleaned_categories-uncleaned_iot"
            elif noise == 4:
                csv_list = [
                    path_to_per_packet_cleaned_categories,
                    path_to_network_noise_cleaned,
                ]
                dataset = combine_csv(csv_list, names)
                name = "cleaned_categories-cleaned_network"
            else:
                csv_list = [
                    path_to_per_packet_cleaned_categories,
                    path_to_network_noise_uncleaned,
                ]
                dataset = combine_csv(csv_list, names)
                name = "cleaned_categories-uncleaned_network"
        else:
            dataset = read_csv(path_to_per_packet_cleaned_categories, names=names)
            name = "cleaned_categories"
    elif experiment_type == 2 and c_uc == 2:
        noise = int(
            input(
                "Select one of the following: \n1. Random \n2. IoT Cleaned \n3. IoT Uncleaned \n"
                "4. Network Cleaned \n5. Network Uncleaned \n6. None\n"
            )
        )
        if not noise in [1, 2, 3, 4, 5, 6]:
            raise ValueError(
                "'noise' selection must be one of the following values: 1, 2, 3, 4, 5 or 6."
            )

        if noise != 6:
            if noise == 1:
                dataset = read_csv(path_to_per_packet_uncleaned_categories, names=names)
                dataset = noise_generator(dataset, names)
                name = "uncleaned_categories-random"
            elif noise == 2:
                csv_list = [
                    path_to_per_packet_uncleaned_categories,
                    path_to_iot_noise_cleaned,
                ]
                dataset = combine_csv(csv_list, names)
                name = "uncleaned_categories-cleaned_iot"
            elif noise == 3:
                csv_list = [
                    path_to_per_packet_uncleaned_categories,
                    path_to_iot_noise_uncleaned,
                ]
                dataset = combine_csv(csv_list, names)
                name = "uncleaned_categories-uncleaned_iot"
            elif noise == 4:
                csv_list = [
                    path_to_per_packet_uncleaned_categories,
                    path_to_network_noise_cleaned,
                ]
                dataset = combine_csv(csv_list, names)
                name = "uncleaned_categories-cleaned_network"
            else:
                csv_list = [
                    path_to_per_packet_uncleaned_categories,
                    path_to_network_noise_uncleaned,
                ]
                dataset = combine_csv(csv_list, names)
                name = "uncleaned_categories-uncleaned_network"
        else:
            dataset = read_csv(path_to_per_packet_uncleaned_categories, names=names)
            name = "uncleaned_categories"
    elif experiment_type == 3:
        device = int(
            input("Select one of the following: \n1. Plug \n2. Bulb \n3. Cam\n")
        )
        if not device in [1, 2, 3]:
            raise ValueError(
                "'device' parameter must be one of the following values: 1, 2 or 3. 1 represents plugs, 2 "
                "represents light bulbs and 3 represents cameras."
            )

        i_ni = int(
            input("Select one of the following: \n1. Interaction \n2. No Interaction\n")
        )
        if not i_ni in [1, 2]:
            raise ValueError(
                "'i_ni' parameter must be one of the following values: 1 or 2."
            )

        if device == 1 and c_uc == 1 and i_ni == 1:
            csv_list = os.listdir(path_to_same_plug_cleaned_interaction)
            csv_list = [
                f"{path_to_same_plug_cleaned_interaction + i}" for i in csv_list
            ]
            dataset = combine_csv(csv_list, names)
            name = "plug-cleaned-interaction"
        elif device == 1 and c_uc == 1 and i_ni == 2:
            csv_list = os.listdir(path_to_same_plug_cleaned_no_interaction)
            csv_list = [
                f"{path_to_same_plug_cleaned_no_interaction + i}" for i in csv_list
            ]
            dataset = combine_csv(csv_list, names)
            name = "plug-cleaned-no_interaction"
        elif device == 1 and c_uc == 2 and i_ni == 1:
            csv_list = os.listdir(path_to_same_plug_uncleaned_interaction)
            csv_list = [
                f"{path_to_same_plug_uncleaned_interaction + i}" for i in csv_list
            ]
            dataset = combine_csv(csv_list, names)
            name = "plug-uncleaned-interaction"
        elif device == 1 and c_uc == 2 and i_ni == 2:
            csv_list = os.listdir(path_to_same_plug_uncleaned_no_interaction)
            csv_list = [
                f"{path_to_same_plug_uncleaned_no_interaction + i}" for i in csv_list
            ]
            dataset = combine_csv(csv_list, names)
            name = "plug-uncleaned-no_interaction"
        elif device == 2 and c_uc == 1 and i_ni == 1:
            csv_list = os.listdir(path_to_same_bulb_cleaned_interaction)
            csv_list = [
                f"{path_to_same_bulb_cleaned_interaction + i}" for i in csv_list
            ]
            dataset = combine_csv(csv_list, names)
            name = "bulb-cleaned-interaction"
        elif device == 2 and c_uc == 1 and i_ni == 2:
            csv_list = os.listdir(path_to_same_bulb_cleaned_no_interaction)
            csv_list = [
                f"{path_to_same_bulb_cleaned_no_interaction + i}" for i in csv_list
            ]
            dataset = combine_csv(csv_list, names)
            name = "bulb-cleaned-no_interaction"
        elif device == 2 and c_uc == 2 and i_ni == 1:
            csv_list = os.listdir(path_to_same_bulb_uncleaned_interaction)
            csv_list = [
                f"{path_to_same_bulb_uncleaned_interaction + i}" for i in csv_list
            ]
            dataset = combine_csv(csv_list, names)
            name = "bulb-uncleaned-interaction"
        elif device == 2 and c_uc == 2 and i_ni == 2:
            csv_list = os.listdir(path_to_same_bulb_uncleaned_no_interaction)
            csv_list = [
                f"{path_to_same_bulb_uncleaned_no_interaction + i}" for i in csv_list
            ]
            dataset = combine_csv(csv_list, names)
            name = "bulb-uncleaned-no_interaction"
        elif device == 3 and c_uc == 1 and i_ni == 1:
            csv_list = os.listdir(path_to_same_cam_cleaned_interaction)
            csv_list = [f"{path_to_same_cam_cleaned_interaction + i}" for i in csv_list]
            dataset = combine_csv(csv_list, names)
            name = "cam-cleaned-interaction"
        elif device == 3 and c_uc == 1 and i_ni == 2:
            csv_list = os.listdir(path_to_same_cam_cleaned_no_interaction)
            csv_list = [
                f"{path_to_same_cam_cleaned_no_interaction + i}" for i in csv_list
            ]
            dataset = combine_csv(csv_list, names)
            name = "cam-cleaned-no_interaction"
        elif device == 3 and c_uc == 2 and i_ni == 1:
            csv_list = os.listdir(path_to_same_cam_uncleaned_interaction)
            csv_list = [
                f"{path_to_same_cam_uncleaned_interaction + i}" for i in csv_list
            ]
            dataset = combine_csv(csv_list, names)
            name = "cam-uncleaned-interaction"
        else:
            csv_list = os.listdir(path_to_same_cam_uncleaned_no_interaction)
            csv_list = [
                f"{path_to_same_cam_uncleaned_no_interaction + i}" for i in csv_list
            ]
            dataset = combine_csv(csv_list, names)
            name = "cam-uncleaned-no_interaction"

    elif experiment_type == 4:
        device = int(
            input("Select one of the following: \n1. Plug \n2. Bulb \n3. Cam\n")
        )
        if not device in [1, 2, 3]:
            raise ValueError(
                "'device' parameter must be one of the following values: 1, 2 or 3. 1 represents plugs, 2 "
                "represents light bulbs and 3 represents cameras."
            )
        if device == 1:
            device_selection = "plug"
        elif device == 2:
            device_selection = "light"
        else:
            device_selection = "cam"

        accum = int(
            input(
                "Select one of the following accumulator sizes: \n128 \n256 \n512 \n1024\n"
            )
        )
        if not accum in [128, 256, 512, 1024]:
            raise ValueError(
                "'accum' parameter must be one of the following values: 128, 256, 512 or 1024."
            )

        names = [f"dim{i}" for i in range(accum)]
        names.append("class")

        window = int(input("Select one of the following window sizes: \n4 \n5 \n6\n"))
        if not window in [4, 5, 6]:
            raise ValueError(
                "'window' parameter must be one of the following values:4, 5, or 6."
            )
        if window == 4:
            combo = int(
                input("Select one of the following combination sizes: \n2 \n3 \n4\n")
            )
        elif window == 5:
            combo = int(
                input(
                    "Select one of the following combination sizes: \n2 \n3 \n4 \n5\n"
                )
            )
        else:
            combo = int(
                input(
                    "Select one of the following combination sizes: \n2 \n3 \n4 \n5 \n6\n"
                )
            )

        if c_uc == 1:
            target_dir = f"{path_to_simhash}{device_selection}/accum_{accum}/window_{window}/combo_{combo}/cleaned/"
            csv_list = os.listdir(target_dir)
            csv_list = [f"{target_dir + i}" for i in csv_list]
            name = f"FlexHash-{device_selection}-accum_{accum}-win_{window}-combo_{combo}-cleaned"
        else:
            target_dir = f"{path_to_simhash}{device_selection}/accum_{accum}/window_{window}/combo_{combo}/uncleaned/"
            csv_list = os.listdir(target_dir)
            csv_list = [f"{target_dir + i}" for i in csv_list]
            name = f"FlexHash-{device_selection}-accum_{accum}-window_{window}-combo_{combo}-uncleaned"

        dataset = combine_csv(csv_list, names)

    elif experiment_type == 5:
        hash_alg = int(
            input(
                "Select on of the following hashing algorithms: \n1. Nilsimsa \n2.FlexHash\n"
            )
        )
        if not hash_alg in [1, 2]:
            raise ValueError(
                "'hash_alg' parameter must be one of the following values: 1 or 2."
            )

        noise = int(
            input(
                "Select one of the following: \n1. Random \n2. IoT Cleaned \n3. IoT Uncleaned \n4. Network "
                "Cleaned \n5. Network Uncleaned\n"
            )
        )
        if not noise in [1, 2, 3, 4, 5]:
            raise ValueError(
                "'noise' parameter must be one of the following values: 1, 2, 3, 4 or 5."
            )

        if hash_alg == 1:
            device = int(
                input("Select one of the following: \n1. Plug \n2. Bulb \n3. Cam\n")
            )
            if not device in [1, 2, 3]:
                raise ValueError(
                    "'device' parameter must be one of the following values: 1, 2 or 3."
                )

            i_ni = int(
                input(
                    "Select one of the following: \n1. Interaction \n2. No Interaction\n"
                )
            )
            if not i_ni in [1, 2]:
                raise ValueError(
                    "'i_ni' parameter must be one of the following values: 1 or 2."
                )

            device_num = int(input("Select a device number (1-8): \n"))
            if not device_num in [1, 2, 3, 4, 5, 6, 7, 8]:
                raise ValueError(
                    "'device_num' parameter must be one of the following values: 1, 2, 3, 4, 5, 6, 7 or 8."
                )

            if device == 1 and c_uc == 1 and i_ni == 1:
                csv_list = sorted(os.listdir(path_to_same_plug_cleaned_interaction))
                csv_list = [
                    path_to_same_plug_cleaned_interaction + csv_list[device_num - 1]
                ]

                name = f"plug-{device_num}-cleaned-interaction_100x"
            elif device == 1 and c_uc == 1 and i_ni == 2:
                csv_list = sorted(os.listdir(path_to_same_plug_cleaned_no_interaction))
                csv_list = [
                    path_to_same_plug_cleaned_no_interaction + csv_list[device_num - 1]
                ]

                name = f"plug-{device_num}-cleaned-no_interaction_100x"
            elif device == 1 and c_uc == 2 and i_ni == 1:
                csv_list = sorted(os.listdir(path_to_same_plug_uncleaned_interaction))
                csv_list = [
                    path_to_same_plug_uncleaned_interaction + csv_list[device_num - 1]
                ]

                name = f"plug-{device_num}-uncleaned-interaction_100x"
            elif device == 1 and c_uc == 2 and i_ni == 2:
                csv_list = sorted(
                    os.listdir(path_to_same_plug_uncleaned_no_interaction)
                )
                csv_list = [
                    path_to_same_plug_uncleaned_no_interaction
                    + csv_list[device_num - 1]
                ]

                name = f"plug-{device_num}-uncleaned-no_interaction_100x"
            elif device == 2 and c_uc == 1 and i_ni == 1:
                csv_list = sorted(os.listdir(path_to_same_bulb_cleaned_interaction))
                csv_list = [
                    path_to_same_bulb_cleaned_interaction + csv_list[device_num - 1]
                ]

                name = f"bulb-{device_num}-cleaned-interaction_100x"
            elif device == 2 and c_uc == 1 and i_ni == 2:
                csv_list = sorted(os.listdir(path_to_same_bulb_cleaned_no_interaction))
                csv_list = [
                    path_to_same_bulb_cleaned_no_interaction + csv_list[device_num - 1]
                ]

                name = f"bulb-{device_num}-cleaned-no_interaction_100x"
            elif device == 2 and c_uc == 2 and i_ni == 1:
                csv_list = sorted(os.listdir(path_to_same_bulb_uncleaned_interaction))
                csv_list = [
                    path_to_same_bulb_uncleaned_interaction + csv_list[device_num - 1]
                ]

                name = f"bulb-{device_num}-uncleaned-interaction_100x"
            elif device == 2 and c_uc == 2 and i_ni == 2:
                csv_list = sorted(
                    os.listdir(path_to_same_bulb_uncleaned_no_interaction)
                )
                csv_list = [
                    path_to_same_bulb_uncleaned_no_interaction
                    + csv_list[device_num - 1]
                ]

                name = f"bulb-{device_num}-uncleaned-no_interaction_100x"
            elif device == 3 and c_uc == 1 and i_ni == 1:
                csv_list = sorted(os.listdir(path_to_same_cam_cleaned_interaction))
                csv_list = [
                    path_to_same_cam_cleaned_interaction + csv_list[device_num - 1]
                ]

                name = f"cam-{device_num}-cleaned-interaction_100x"
            elif device == 3 and c_uc == 1 and i_ni == 2:
                csv_list = sorted(os.listdir(path_to_same_cam_cleaned_no_interaction))
                csv_list = [
                    path_to_same_cam_cleaned_no_interaction + csv_list[device_num - 1]
                ]

                name = f"cam-{device_num}-cleaned-no_interaction_100x"
            elif device == 3 and c_uc == 2 and i_ni == 1:
                csv_list = sorted(os.listdir(path_to_same_cam_uncleaned_interaction))
                csv_list = [
                    path_to_same_cam_uncleaned_interaction + csv_list[device_num - 1]
                ]

                name = f"cam-{device_num}-uncleaned-interaction_100x"
            else:
                csv_list = sorted(os.listdir(path_to_same_cam_uncleaned_no_interaction))
                csv_list = [
                    path_to_same_cam_uncleaned_no_interaction + csv_list[device_num - 1]
                ]

                name = f"cam-{device_num}-uncleaned-no_interaction_100x"

            if noise == 1:
                dataset = combine_csv(csv_list, names)
                dataset = noise_generator(dataset, names, False)
                name += "-random"
            if noise == 2:
                csv_list.append(path_to_iot_noise_cleaned)
                name += "-cleaned_iot"
                dataset = combine_csv(csv_list, names)
            elif noise == 3:
                csv_list.append(path_to_iot_noise_uncleaned)
                name += "-uncleaned_iot"
                dataset = combine_csv(csv_list, names)
            elif noise == 4:
                csv_list.append(path_to_network_noise_cleaned)
                name += "-cleaned_network"
                dataset = combine_csv(csv_list, names)
            else:
                csv_list.append(path_to_network_noise_uncleaned)
                name += "-uncleaned_network"
                dataset = combine_csv(csv_list, names)

    else:
        if experiment_type == 6 and c_uc == 1:
            noise = int(
                input(
                    "Select one of the following: \n1. Random \n2. IoT Cleaned \n3. IoT Uncleaned \n"
                    "4. Network Cleaned \n5. Network Uncleaned \n6. None\n"
                )
            )
            if not noise in [1, 2, 3, 4, 5, 6]:
                raise ValueError(
                    "'noise' selection must be one of the following values: 1, 2, 3, 4, 5 or 6."
                )

            if noise != 6:
                if noise == 1:
                    dataset = read_csv(
                        path_to_per_packet_cleaned_categories, names=names
                    )
                    dataset = noise_generator(dataset, names)
                    name = "cleaned_categories-random"
                elif noise == 2:
                    csv_list = [
                        path_to_per_packet_cleaned_categories,
                        path_to_iot_noise_cleaned,
                    ]
                    dataset = combine_csv(csv_list, names)
                    name = "cleaned_categories-cleaned_iot"
                elif noise == 3:
                    csv_list = [
                        path_to_per_packet_cleaned_categories,
                        path_to_iot_noise_uncleaned,
                    ]
                    dataset = combine_csv(csv_list, names)
                    name = "cleaned_categories-uncleaned_iot"
                elif noise == 4:
                    csv_list = [
                        path_to_per_packet_cleaned_categories,
                        path_to_network_noise_cleaned,
                    ]
                    dataset = combine_csv(csv_list, names)
                    name = "cleaned_categories-cleaned_network"
                else:
                    csv_list = [
                        path_to_per_packet_cleaned_categories,
                        path_to_network_noise_uncleaned,
                    ]
                    dataset = combine_csv(csv_list, names)
                    name = "cleaned_categories-uncleaned_network"
            else:
                dataset = read_csv(path_to_per_packet_cleaned_categories, names=names)
                name = "cleaned_PvLvC"
        elif experiment_type == 6 and c_uc == 2:
            noise = int(
                input(
                    "Select one of the following: \n1. Random \n2. IoT Cleaned \n3. IoT Uncleaned \n"
                    "4. Network Cleaned \n5. Network Uncleaned \n6. None\n"
                )
            )
            if not noise in [1, 2, 3, 4, 5, 6]:
                raise ValueError(
                    "'noise' selection must be one of the following values: 1, 2, 3, 4, 5 or 6."
                )

            if noise != 6:
                if noise == 1:
                    dataset = read_csv(
                        path_to_per_packet_uncleaned_categories, names=names
                    )
                    dataset = noise_generator(dataset, names)
                    name = "uncleaned_categories-random"
                elif noise == 2:
                    csv_list = [
                        path_to_per_packet_uncleaned_categories,
                        path_to_iot_noise_cleaned,
                    ]
                    dataset = combine_csv(csv_list, names)
                    name = "uncleaned_categories-cleaned_iot"
                elif noise == 3:
                    csv_list = [
                        path_to_per_packet_uncleaned_categories,
                        path_to_iot_noise_uncleaned,
                    ]
                    dataset = combine_csv(csv_list, names)
                    name = "uncleaned_categories-uncleaned_iot"
                elif noise == 4:
                    csv_list = [
                        path_to_per_packet_uncleaned_categories,
                        path_to_network_noise_cleaned,
                    ]
                    dataset = combine_csv(csv_list, names)
                    name = "uncleaned_categories-cleaned_network"
                else:
                    csv_list = [
                        path_to_per_packet_uncleaned_categories,
                        path_to_network_noise_uncleaned,
                    ]
                    dataset = combine_csv(csv_list, names)
                    name = "uncleaned_categories-uncleaned_network"
            else:
                dataset = read_csv(path_to_per_packet_uncleaned_categories, names=names)
                name = "uncleaned_PvLvC"

    return dataset, name


def get_dataset_parameterized(
    experiment_type=1, c_uc=1, noise=1, device=1, i_ni=1, accum=128, window=4, combo=2
):
    names = [
        "dim1",
        "dim2",
        "dim3",
        "dim4",
        "dim5",
        "dim6",
        "dim7",
        "dim8",
        "dim9",
        "dim10",
        "dim11",
        "dim12",
        "dim13",
        "dim14",
        "dim15",
        "dim16",
        "dim17",
        "dim18",
        "dim19",
        "dim20",
        "dim21",
        "dim22",
        "dim23",
        "dim24",
        "dim25",
        "dim26",
        "dim27",
        "dim28",
        "dim29",
        "dim30",
        "dim31",
        "dim32",
        "class",
    ]

    if experiment_type == 1 and c_uc == 1:
        if noise != 6:
            if noise == 1:
                dataset = read_csv(path_to_per_packet_cleaned_devices, names=names)
                dataset = noise_generator(dataset, names)
                name = "cleaned_devices-random"
            elif noise == 2:
                csv_list = [
                    path_to_per_packet_cleaned_devices,
                    path_to_iot_noise_cleaned,
                ]
                dataset = combine_csv(csv_list, names)
                name = "cleaned_devices-cleaned_iot"
            elif noise == 3:
                csv_list = [
                    path_to_per_packet_cleaned_devices,
                    path_to_iot_noise_uncleaned,
                ]
                dataset = combine_csv(csv_list, names)
                name = "cleaned_devices-uncleaned_iot"
            elif noise == 4:
                csv_list = [
                    path_to_per_packet_cleaned_devices,
                    path_to_network_noise_cleaned,
                ]
                dataset = combine_csv(csv_list, names)
                name = "cleaned_devices-cleaned_network"
            elif noise == 5:
                csv_list = [
                    path_to_per_packet_cleaned_devices,
                    path_to_network_noise_uncleaned,
                ]
                dataset = combine_csv(csv_list, names)
                name = "cleaned_devices-uncleaned_network"
        else:
            dataset = read_csv(path_to_per_packet_cleaned_devices, names=names)
            name = "cleaned_devices"
    elif experiment_type == 1 and c_uc == 2:
        if noise != 6:
            if noise == 1:
                dataset = read_csv(path_to_per_packet_uncleaned_devices, names=names)
                dataset = noise_generator(dataset, names)
                name = "uncleaned_devices-random"
            elif noise == 2:
                csv_list = [
                    path_to_per_packet_uncleaned_devices,
                    path_to_iot_noise_cleaned,
                ]
                dataset = combine_csv(csv_list, names)
                name = "uncleaned_devices-cleaned_iot"
            elif noise == 3:
                csv_list = [
                    path_to_per_packet_uncleaned_devices,
                    path_to_iot_noise_uncleaned,
                ]
                dataset = combine_csv(csv_list, names)
                name = "uncleaned_devices-uncleaned_iot"
            elif noise == 4:
                csv_list = [
                    path_to_per_packet_uncleaned_devices,
                    path_to_network_noise_cleaned,
                ]
                dataset = combine_csv(csv_list, names)
                name = "uncleaned_devices-cleaned_network"
            elif noise == 5:
                csv_list = [
                    path_to_per_packet_uncleaned_devices,
                    path_to_network_noise_uncleaned,
                ]
                dataset = combine_csv(csv_list, names)
                name = "uncleaned_devices-uncleaned-network"
        else:
            dataset = read_csv(path_to_per_packet_uncleaned_devices, names=names)
            name = "uncleaned_devices"
    elif experiment_type == 2 and c_uc == 1:
        if noise != 6:
            if noise == 1:
                dataset = read_csv(path_to_per_packet_cleaned_categories, names=names)
                dataset = noise_generator(dataset, names)
                name = "cleaned_categories-random"
            elif noise == 2:
                csv_list = [
                    path_to_per_packet_cleaned_categories,
                    path_to_iot_noise_cleaned,
                ]
                dataset = combine_csv(csv_list, names)
                name = "cleaned_categories-cleaned_iot"
            elif noise == 3:
                csv_list = [
                    path_to_per_packet_cleaned_categories,
                    path_to_iot_noise_uncleaned,
                ]
                dataset = combine_csv(csv_list, names)
                name = "cleaned_categories-uncleaned_iot"
            elif noise == 4:
                csv_list = [
                    path_to_per_packet_cleaned_categories,
                    path_to_network_noise_cleaned,
                ]
                dataset = combine_csv(csv_list, names)
                name = "cleaned_categories-cleaned_network"
            elif noise == 5:
                csv_list = [
                    path_to_per_packet_cleaned_categories,
                    path_to_network_noise_uncleaned,
                ]
                dataset = combine_csv(csv_list, names)
                name = "cleaned_categories-uncleaned_network"
        else:
            dataset = read_csv(path_to_per_packet_cleaned_categories, names=names)
            name = "cleaned_categories"
    elif experiment_type == 2 and c_uc == 2:
        if noise != 6:
            if noise == 1:
                dataset = read_csv(path_to_per_packet_uncleaned_categories, names=names)
                dataset = noise_generator(dataset, names)
                name = "uncleaned_categories-random"
            elif noise == 2:
                csv_list = [
                    path_to_per_packet_uncleaned_categories,
                    path_to_iot_noise_cleaned,
                ]
                dataset = combine_csv(csv_list, names)
                name = "uncleaned_categories-cleaned_iot"
            elif noise == 3:
                csv_list = [
                    path_to_per_packet_uncleaned_categories,
                    path_to_iot_noise_uncleaned,
                ]
                dataset = combine_csv(csv_list, names)
                name = "uncleaned_categories-uncleaned_iot"
            elif noise == 4:
                csv_list = [
                    path_to_per_packet_uncleaned_categories,
                    path_to_network_noise_cleaned,
                ]
                dataset = combine_csv(csv_list, names)
                name = "uncleaned_categories-cleaned_network"
            elif noise == 5:
                csv_list = [
                    path_to_per_packet_uncleaned_categories,
                    path_to_network_noise_uncleaned,
                ]
                dataset = combine_csv(csv_list, names)
                name = "uncleaned_categories-uncleaned_network"
        else:
            dataset = read_csv(path_to_per_packet_uncleaned_categories, names=names)
            name = "uncleaned_categories"
    elif experiment_type == 3:
        if not device in [1, 2, 3]:
            raise ValueError(
                "'device' parameter must be one of the following values: 1, 2 or 3. 1 represents plugs, 2 "
                "represents light bulbs and 3 represents cameras."
            )

        if device == 1 and c_uc == 1 and i_ni == 1:
            csv_list = os.listdir(path_to_same_plug_cleaned_interaction)
            csv_list = [
                f"{path_to_same_plug_cleaned_interaction + i}" for i in csv_list
            ]
            dataset = combine_csv(csv_list, names)
            name = "plug-cleaned-interaction"
        elif device == 1 and c_uc == 1 and i_ni == 2:
            csv_list = os.listdir(path_to_same_plug_cleaned_no_interaction)
            csv_list = [
                f"{path_to_same_plug_cleaned_no_interaction + i}" for i in csv_list
            ]
            dataset = combine_csv(csv_list, names)
            name = "plug-cleaned-no_interaction"
        elif device == 1 and c_uc == 2 and i_ni == 1:
            csv_list = os.listdir(path_to_same_plug_uncleaned_interaction)
            csv_list = [
                f"{path_to_same_plug_uncleaned_interaction + i}" for i in csv_list
            ]
            dataset = combine_csv(csv_list, names)
            name = "plug-uncleaned-interaction"
        elif device == 1 and c_uc == 2 and i_ni == 2:
            csv_list = os.listdir(path_to_same_plug_uncleaned_no_interaction)
            csv_list = [
                f"{path_to_same_plug_uncleaned_no_interaction + i}" for i in csv_list
            ]
            dataset = combine_csv(csv_list, names)
            name = "plug-uncleaned-no_interaction"
        elif device == 2 and c_uc == 1 and i_ni == 1:
            csv_list = os.listdir(path_to_same_bulb_cleaned_interaction)
            csv_list = [
                f"{path_to_same_bulb_cleaned_interaction + i}" for i in csv_list
            ]
            dataset = combine_csv(csv_list, names)
            name = "bulb-cleaned-interaction"
        elif device == 2 and c_uc == 1 and i_ni == 2:
            csv_list = os.listdir(path_to_same_bulb_cleaned_no_interaction)
            csv_list = [
                f"{path_to_same_bulb_cleaned_no_interaction + i}" for i in csv_list
            ]
            dataset = combine_csv(csv_list, names)
            name = "bulb-cleaned-no_interaction"
        elif device == 2 and c_uc == 2 and i_ni == 1:
            csv_list = os.listdir(path_to_same_bulb_uncleaned_interaction)
            csv_list = [
                f"{path_to_same_bulb_uncleaned_interaction + i}" for i in csv_list
            ]
            dataset = combine_csv(csv_list, names)
            name = "bulb-uncleaned-interaction"
        elif device == 2 and c_uc == 2 and i_ni == 2:
            csv_list = os.listdir(path_to_same_bulb_uncleaned_no_interaction)
            csv_list = [
                f"{path_to_same_bulb_uncleaned_no_interaction + i}" for i in csv_list
            ]
            dataset = combine_csv(csv_list, names)
            name = "bulb-uncleaned-no_interaction"
        elif device == 3 and c_uc == 1 and i_ni == 1:
            csv_list = os.listdir(path_to_same_cam_cleaned_interaction)
            csv_list = [f"{path_to_same_cam_cleaned_interaction + i}" for i in csv_list]
            dataset = combine_csv(csv_list, names)
            name = "cam-cleaned-interaction"
        elif device == 3 and c_uc == 1 and i_ni == 2:
            csv_list = os.listdir(path_to_same_cam_cleaned_no_interaction)
            csv_list = [
                f"{path_to_same_cam_cleaned_no_interaction + i}" for i in csv_list
            ]
            dataset = combine_csv(csv_list, names)
            name = "cam-cleaned-no_interaction"
        elif device == 3 and c_uc == 2 and i_ni == 1:
            csv_list = os.listdir(path_to_same_cam_uncleaned_interaction)
            csv_list = [
                f"{path_to_same_cam_uncleaned_interaction + i}" for i in csv_list
            ]
            dataset = combine_csv(csv_list, names)
            name = "cam-uncleaned-interaction"
        elif device == 3 and c_uc == 2 and i_ni == 2:
            csv_list = os.listdir(path_to_same_cam_uncleaned_no_interaction)
            csv_list = [
                f"{path_to_same_cam_uncleaned_no_interaction + i}" for i in csv_list
            ]
            dataset = combine_csv(csv_list, names)
            name = "cam-uncleaned-no_interaction"

    elif experiment_type == 4:
        if not device in [1, 2, 3]:
            raise ValueError(
                "'device' parameter must be one of the following values: 1, 2 or 3. 1 represents plugs, 2 "
                "represents light bulbs and 3 represents cameras."
            )
        if device == 1:
            device_selection = "plug"
        elif device == 2:
            device_selection = "light"
        else:
            device_selection = "cam"

        path_to_simhash = (
            f"/home/nthom/Documents/SmartRecon/Fingerprinting in Noisy Network Environments/data/simhashes/"
            f"{device_selection}/"
        )

        if accum == 128:
            names = [
                "dim1",
                "dim2",
                "dim3",
                "dim4",
                "dim5",
                "dim6",
                "dim7",
                "dim8",
                "dim9",
                "dim10",
                "dim11",
                "dim12",
                "dim13",
                "dim14",
                "dim15",
                "dim16",
                "class",
            ]
        elif accum == 256:
            names = [
                "dim1",
                "dim2",
                "dim3",
                "dim4",
                "dim5",
                "dim6",
                "dim7",
                "dim8",
                "dim9",
                "dim10",
                "dim11",
                "dim12",
                "dim13",
                "dim14",
                "dim15",
                "dim16",
                "dim17",
                "dim18",
                "dim19",
                "dim20",
                "dim21",
                "dim22",
                "dim23",
                "dim24",
                "dim25",
                "dim26",
                "dim27",
                "dim28",
                "dim29",
                "dim30",
                "dim31",
                "dim32",
                "class",
            ]
        elif accum == 512:
            names = [
                "dim1",
                "dim2",
                "dim3",
                "dim4",
                "dim5",
                "dim6",
                "dim7",
                "dim8",
                "dim9",
                "dim10",
                "dim11",
                "dim12",
                "dim13",
                "dim14",
                "dim15",
                "dim16",
                "dim17",
                "dim18",
                "dim19",
                "dim20",
                "dim21",
                "dim22",
                "dim23",
                "dim24",
                "dim25",
                "dim26",
                "dim27",
                "dim28",
                "dim29",
                "dim30",
                "dim31",
                "dim32",
                "dim33",
                "dim34",
                "dim35",
                "dim36",
                "dim37",
                "dim38",
                "dim39",
                "dim40",
                "dim41",
                "dim42",
                "dim43",
                "dim44",
                "dim45",
                "dim46",
                "dim47",
                "dim48",
                "dim49",
                "dim50",
                "dim51",
                "dim52",
                "dim53",
                "dim54",
                "dim55",
                "dim56",
                "dim57",
                "dim58",
                "dim59",
                "dim60",
                "dim61",
                "dim62",
                "dim63",
                "dim64",
                "class",
            ]
        if accum == 1024:
            names = [
                "dim1",
                "dim2",
                "dim3",
                "dim4",
                "dim5",
                "dim6",
                "dim7",
                "dim8",
                "dim9",
                "dim10",
                "dim11",
                "dim12",
                "dim13",
                "dim14",
                "dim15",
                "dim16",
                "dim17",
                "dim18",
                "dim19",
                "dim20",
                "dim21",
                "dim22",
                "dim23",
                "dim24",
                "dim25",
                "dim26",
                "dim27",
                "dim28",
                "dim29",
                "dim30",
                "dim31",
                "dim32",
                "dim33",
                "dim34",
                "dim35",
                "dim36",
                "dim37",
                "dim38",
                "dim39",
                "dim40",
                "dim41",
                "dim42",
                "dim43",
                "dim44",
                "dim45",
                "dim46",
                "dim47",
                "dim48",
                "dim49",
                "dim50",
                "dim51",
                "dim52",
                "dim53",
                "dim54",
                "dim55",
                "dim56",
                "dim57",
                "dim58",
                "dim59",
                "dim60",
                "dim61",
                "dim62",
                "dim63",
                "dim64",
                "dim65",
                "dim66",
                "dim67",
                "dim68",
                "dim69",
                "dim70",
                "dim71",
                "dim72",
                "dim73",
                "dim74",
                "dim75",
                "dim76",
                "dim77",
                "dim78",
                "dim79",
                "dim80",
                "dim81",
                "dim82",
                "dim83",
                "dim84",
                "dim85",
                "dim86",
                "dim87",
                "dim88",
                "dim89",
                "dim90",
                "dim91",
                "dim92",
                "dim93",
                "dim94",
                "dim95",
                "dim96",
                "dim97",
                "dim98",
                "dim99",
                "dim100",
                "dim101",
                "dim102",
                "dim103",
                "dim104",
                "dim105",
                "dim106",
                "dim107",
                "dim108",
                "dim109",
                "dim110",
                "dim111",
                "dim112",
                "dim113",
                "dim114",
                "dim115",
                "dim116",
                "dim117",
                "dim118",
                "dim119",
                "dim120",
                "dim121",
                "dim122",
                "dim123",
                "dim124",
                "dim125",
                "dim126",
                "dim127",
                "dim128",
                "class",
            ]

        if c_uc == 1:
            target_dir = (
                f"{path_to_simhash}accum_{accum}/window_{window}/combo_{combo}/cleaned/"
            )
            csv_list = os.listdir(target_dir)
            csv_list = [f"{target_dir + i}" for i in csv_list]
            name = f"FlexHash-{device_selection}-accum_{accum}-window_{window}-combo_{combo}-cleaned"
        elif c_uc == 2:
            target_dir = f"{path_to_simhash}accum_{accum}/window_{window}/combo_{combo}/uncleaned/"
            csv_list = os.listdir(target_dir)
            csv_list = [f"{target_dir + i}" for i in csv_list]
            name = f"FlexHash-{device_selection}-accum_{accum}-window_{window}-combo_{combo}-uncleaned"

        dataset = combine_csv(csv_list, names)
    return dataset, name


def get_dataset_parameterized_auto_select(
    device, device_c_uc, noise, noise_c_uc, accum, window, combo
):
    if not accum in [128, 256, 512, 1024]:
        raise ValueError(
            "'accum' parameter must be one of the following values: 128, 256, 512 or 1024."
        )
    if accum == 128:
        names = [
            "dim1",
            "dim2",
            "dim3",
            "dim4",
            "dim5",
            "dim6",
            "dim7",
            "dim8",
            "dim9",
            "dim10",
            "dim11",
            "dim12",
            "dim13",
            "dim14",
            "dim15",
            "dim16",
            "class",
        ]
    elif accum == 256:
        names = [
            "dim1",
            "dim2",
            "dim3",
            "dim4",
            "dim5",
            "dim6",
            "dim7",
            "dim8",
            "dim9",
            "dim10",
            "dim11",
            "dim12",
            "dim13",
            "dim14",
            "dim15",
            "dim16",
            "dim17",
            "dim18",
            "dim19",
            "dim20",
            "dim21",
            "dim22",
            "dim23",
            "dim24",
            "dim25",
            "dim26",
            "dim27",
            "dim28",
            "dim29",
            "dim30",
            "dim31",
            "dim32",
            "class",
        ]
    elif accum == 512:
        names = [
            "dim1",
            "dim2",
            "dim3",
            "dim4",
            "dim5",
            "dim6",
            "dim7",
            "dim8",
            "dim9",
            "dim10",
            "dim11",
            "dim12",
            "dim13",
            "dim14",
            "dim15",
            "dim16",
            "dim17",
            "dim18",
            "dim19",
            "dim20",
            "dim21",
            "dim22",
            "dim23",
            "dim24",
            "dim25",
            "dim26",
            "dim27",
            "dim28",
            "dim29",
            "dim30",
            "dim31",
            "dim32",
            "dim33",
            "dim34",
            "dim35",
            "dim36",
            "dim37",
            "dim38",
            "dim39",
            "dim40",
            "dim41",
            "dim42",
            "dim43",
            "dim44",
            "dim45",
            "dim46",
            "dim47",
            "dim48",
            "dim49",
            "dim50",
            "dim51",
            "dim52",
            "dim53",
            "dim54",
            "dim55",
            "dim56",
            "dim57",
            "dim58",
            "dim59",
            "dim60",
            "dim61",
            "dim62",
            "dim63",
            "dim64",
            "class",
        ]
    else:
        names = [
            "dim1",
            "dim2",
            "dim3",
            "dim4",
            "dim5",
            "dim6",
            "dim7",
            "dim8",
            "dim9",
            "dim10",
            "dim11",
            "dim12",
            "dim13",
            "dim14",
            "dim15",
            "dim16",
            "dim17",
            "dim18",
            "dim19",
            "dim20",
            "dim21",
            "dim22",
            "dim23",
            "dim24",
            "dim25",
            "dim26",
            "dim27",
            "dim28",
            "dim29",
            "dim30",
            "dim31",
            "dim32",
            "dim33",
            "dim34",
            "dim35",
            "dim36",
            "dim37",
            "dim38",
            "dim39",
            "dim40",
            "dim41",
            "dim42",
            "dim43",
            "dim44",
            "dim45",
            "dim46",
            "dim47",
            "dim48",
            "dim49",
            "dim50",
            "dim51",
            "dim52",
            "dim53",
            "dim54",
            "dim55",
            "dim56",
            "dim57",
            "dim58",
            "dim59",
            "dim60",
            "dim61",
            "dim62",
            "dim63",
            "dim64",
            "dim65",
            "dim66",
            "dim67",
            "dim68",
            "dim69",
            "dim70",
            "dim71",
            "dim72",
            "dim73",
            "dim74",
            "dim75",
            "dim76",
            "dim77",
            "dim78",
            "dim79",
            "dim80",
            "dim81",
            "dim82",
            "dim83",
            "dim84",
            "dim85",
            "dim86",
            "dim87",
            "dim88",
            "dim89",
            "dim90",
            "dim91",
            "dim92",
            "dim93",
            "dim94",
            "dim95",
            "dim96",
            "dim97",
            "dim98",
            "dim99",
            "dim100",
            "dim101",
            "dim102",
            "dim103",
            "dim104",
            "dim105",
            "dim106",
            "dim107",
            "dim108",
            "dim109",
            "dim110",
            "dim111",
            "dim112",
            "dim113",
            "dim114",
            "dim115",
            "dim116",
            "dim117",
            "dim118",
            "dim119",
            "dim120",
            "dim121",
            "dim122",
            "dim123",
            "dim124",
            "dim125",
            "dim126",
            "dim127",
            "dim128",
            "class",
        ]

    if not device in [1, 2, 3]:
        raise ValueError(
            "'device' parameter must be one of the following values: 1, 2 or 3. 1 represents plugs, 2 "
            "represents light bulbs and 3 represents cameras."
        )
    if device == 1:
        device_selection = "plug"
    elif device == 2:
        device_selection = "light"
    else:
        device_selection = "cam"

    if not device_c_uc in [1, 2]:
        raise ValueError(
            "'device_c_uc' parameter must be one of the following values: 1 or 2."
        )
    if device_c_uc == 1:
        device_c_uc_selection = "cleaned"
    else:
        device_c_uc_selection = "uncleaned"

    path_to_simhash = (
        f"/home/nthom/Documents/SmartRecon/Fingerprinting in Noisy Network Environments/data/simhashes/"
        f"{device_selection}/accum_{accum}/window_{window}/combo_{combo}/{device_c_uc_selection}/"
    )
    device_csv_list = os.listdir(path_to_simhash)
    device_csv_list = [f"{path_to_simhash + i}" for i in device_csv_list]

    if not noise in [1, 2, 3, 4]:
        raise ValueError(
            "'noise' parameter must be one of the following values: 1, 2, 3 or 4."
        )
    if noise not in [3, 4]:
        if not noise_c_uc in [1, 2]:
            raise ValueError(
                "'noise_c_uc' parameter must be one of the following values: 1 or 2."
            )
        if noise_c_uc == 1:
            noise_c_uc_selection = "cleaned"
        else:
            noise_c_uc_selection = "uncleaned"

        if noise == 1:
            noise_selection = "iot"
        else:
            noise_selection = "network"

        path_to_noise = (
            f"/home/nthom/Documents/SmartRecon/Fingerprinting in Noisy Network Environments/data/"
            f"FlexHash_noise/{noise_selection}/accum_{accum}/window_{window}/combo_{combo}/"
            f"{noise_c_uc_selection}/"
        )
        noise_csv_list = os.listdir(path_to_noise)
        noise_csv_list = [f"{path_to_noise + i}" for i in noise_csv_list]

        csv_list = device_csv_list + noise_csv_list
        dataset = combine_csv(csv_list, names)

        noise_name = f"{noise_selection}_{noise_c_uc_selection}"
    elif noise == 4:
        noise_csv_list = []
        noise_name = "none"
        csv_list = device_csv_list + noise_csv_list
        dataset = combine_csv(csv_list, names)
    else:
        noise_name = "random"
        dataset = combine_csv(device_csv_list, names)
        dataset = noise_generator(dataset, names)

    name = (
        f"FlexHash-{device_selection}_{device_c_uc_selection}-{noise_name}-"
        f"accum_{accum}-window_{window}-combo_{combo}"
    )

    return dataset, name
