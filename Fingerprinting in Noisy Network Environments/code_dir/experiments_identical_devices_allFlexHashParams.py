import gc
import os
from socket import gethostname

import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from autogluon.tabular import TabularDataset, TabularPredictor

from utils import combine_csv, combine_csv_category

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

c_uc_list = [1]
device_list = [1, 2, 3]
accum_list = [128, 256, 512, 1024]
window_list = [4, 5, 6]

for c_uc in c_uc_list:
    if c_uc == 1:
        c_uc = "cleaned"
    else:
        c_uc = "uncleaned"

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

                    name_of_current_data = f"FlexHash-identicalDeviceAllFlexHashParams-{device}-accum_{accum}-window_" \
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

                    names = list(range(x_train.shape[1]))
                    train_dataset_df = pd.DataFrame(x_train, columns=names)
                    train_dataset_df.insert(train_dataset_df.shape[1], "class", y_train)

                    names = list(range(x_test.shape[1]))
                    test_dataset_df = pd.DataFrame(x_test, columns=names)
                    test_dataset_df.insert(test_dataset_df.shape[1], "class", y_test)

                    del (
                        x,
                        y,
                        y_original,
                        x_train,
                        y_train,
                        x_test,
                        y_test,
                        names,
                        dataset,
                        y_temp
                    )

                    model_save_path = f"agModels-{name_of_current_data}_{gethostname()}"

                    train_dataset_td = TabularDataset(train_dataset_df)
                    label = "class"
                    print("Summary of class variable: \n", train_dataset_td[label].describe())

                    predictor = TabularPredictor(
                        eval_metric="f1_micro", label="class", path=model_save_path
                    ).fit(train_dataset_td, presets="best_quality",
                          excluded_model_types=["CAT", "KNN", "LR", "NN_TORCH", "AG_AUTOMM"],
                          keep_only_best=True
                          )

                    results = predictor.fit_summary()

                    predictor = TabularPredictor.load(model_save_path)

                    test_dataset_td = TabularDataset(test_dataset_df)
                    y_test = test_dataset_td[label]
                    test_data_noLabel = test_dataset_td.drop(columns=[label])

                    y_pred = predictor.predict(test_data_noLabel)
                    perf = predictor.evaluate_predictions(
                        y_true=y_test, y_pred=y_pred, auxiliary_metrics=True
                    )

                    leaderboard_df = predictor.leaderboard(test_dataset_td)
                    leaderboard_df.to_csv(
                        f"agLeaderboard_{name_of_current_data}_{gethostname()}.csv"
                    )
