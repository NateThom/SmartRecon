import gc
import os
from socket import gethostname
import shutil

import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from autogluon.tabular import TabularDataset, TabularPredictor

from utils import combine_csv

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

path_to_simhash = "/home/nthom/Documents/SmartRecon/Fingerprinting in Noisy Network Environments/data/simhashes/"

device_list = int(input("Select one of the following: \n1. Plug \n2. Light \n3. Cam \n4. All\n"))
if not device_list in [1, 2, 3, 4]:
    raise ValueError(
        "'device_list' selection must be one of the following values: 1 or 2,"
    )
if device_list == 1:
    device_list = [1]
elif device_list == 2:
    device_list = [2]
elif device_list == 3:
    device_list = [3]
else:
    device_list = [1, 2, 3]

accum_list = int(input("Select one of the following: \n128 \n256 \n512 \n1024\n"))
if not accum_list in [128, 256, 512, 1024]:
    raise ValueError(
        "'device_list' selection must be one of the following values: 1 or 2,"
    )
accum_list = [accum_list]
# accum_list = [128, 256, 512, 1024]

window_list = int(input("Select one of the following: \n4 \n5 \n6 \n"))
if not window_list in [4, 5, 6]:
    raise ValueError(
        "'window_list' selection must be one of the following values: 4, 5, or 6"
    )
window_list = [window_list]
# window_list = [4, 5, 6]

all_params_best_model_performance = [["Device", "Accum", "Window", "Combo", "Model Name", "f1_micro", "accuracy",
                                      "balanced_accuracy", "mcc"]]
c_uc = "cleaned"
for device in device_list:
    if device == 1:
        device = "plug"
    elif device == 2:
        device = "light"
    elif device == 3:
        device = "cam"

    for accum in accum_list:
        names = [f"dim{i}" for i in range(accum // 8)]
        names.append("class")

        for window in window_list:
            if window == 4:
                combo_list = [2, 3, 4]
            elif window == 5:
                combo_list = [2, 3, 4, 5]
            else:
                combo_list = [2, 3, 4, 5, 6]

            for combo in combo_list:
                csv_list = []

                target_dir = f"{path_to_simhash}{device}/accum_{accum}/window_{window}/combo_{combo}/{c_uc}/"
                for j in os.listdir(target_dir):
                    csv_list.append(target_dir + j)

                name_of_current_data = f"FlexHash-identical{device}AllFlexHashParams-{device}-accum_{accum}-window_" \
                                       f"{window}-combo_{combo}-cleaned"
                dataset = combine_csv(csv_list, names)
                dataset.reset_index(drop=True, inplace=True)

                print(f"*** Total samples in {name_of_current_data}: {len(dataset.index)} ***")
                for device_name in sorted(dataset["class"].unique()):
                    num_samples = len((dataset[dataset["class"] == device_name]).index)
                    print(
                        f"*** Samples for device: {device_name} in {name_of_current_data}: {num_samples} "
                        f"({num_samples/dataset.shape[0]}%) ***"
                    )

                # x is the entire dataframe except for the class column
                x = dataset.drop(["class"], axis=1)

                # y_original is an unaltered list of all values in the class column
                y_original = dataset["class"].values.tolist()

                # y is a dataframe of only the class column and the values have been converted to numeric
                # representation
                counter = 0
                y_temp = dataset["class"].tolist()
                for unique_value in sorted(dataset["class"].unique()):
                    for index, value in enumerate(dataset["class"]):
                        if value == unique_value:
                            y_temp[index] = counter
                    counter += 1
                dataset["class"] = y_temp

                y = dataset["class"]

                x_train, x_test, y_train, y_test = train_test_split(
                    x.values, y.values, test_size=0.2, stratify=y.values
                )

                col_names = list(range(x_train.shape[1]))
                train_dataset_df = pd.DataFrame(x_train, columns=col_names)
                train_dataset_df.insert(train_dataset_df.shape[1], "class", y_train)

                col_names = list(range(x_test.shape[1]))
                test_dataset_df = pd.DataFrame(x_test, columns=col_names)
                test_dataset_df.insert(test_dataset_df.shape[1], "class", y_test)

                del (
                    x,
                    y,
                    y_original,
                    x_train,
                    y_train,
                    x_test,
                    y_test,
                    dataset,
                    y_temp
                )

                model_save_path = f"agModels-{name_of_current_data}_{gethostname()}"

                train_dataset_td = TabularDataset(train_dataset_df)

                # subsample_size = 1600
                # train_dataset_td = train_dataset_td.sample(n=subsample_size, random_state=0)

                label = "class"
                print("Summary of class variable: \n", train_dataset_td[label].describe())

                del (train_dataset_df)

                predictor = TabularPredictor(
                    eval_metric="f1_micro", label="class", path=model_save_path
                ).fit(train_dataset_td, presets="best_quality",
                      excluded_model_types=["CAT", "KNN", "RF", "FASTAI", "LR", "NN_TORCH", "AG_AUTOMM"],
                      keep_only_best=True
                      )

                results = predictor.fit_summary()

                test_dataset_td = TabularDataset(test_dataset_df)

                # test_dataset_td       = test_dataset_td.sample(n=subsample_size, random_state=0)

                del (train_dataset_td, test_dataset_df)

                best_model_name = predictor.get_model_best()
                best_model_scores = predictor.evaluate(test_dataset_td)

                best_model_output_list = [device, accum, window, combo, best_model_name]
                for key in best_model_scores.keys():
                    best_model_output_list.append(best_model_scores[key])

                all_params_best_model_performance.append(best_model_output_list)

                # print(f"\n\n######\n{best_model_output_list}\n#####\n")

                # predictor.save_space()

                shutil.rmtree(f"agModels-{name_of_current_data}_{gethostname()}")

                output_df = pd.DataFrame(all_params_best_model_performance)
                output_df.to_csv(f"{device}_{accum}_{window}_allFlexHashParams_{gethostname()}.csv", index=False)

            # break
        # break
    # break