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

device_list = ["plug", "light", "cam"]

c_uc = int(input("Select one of the following: \n1. Cleaned \n2. Uncleaned\n"))
if not c_uc in [1, 2]:
    raise ValueError(
        "'c_uc' selection must be one of the following values: 1 or 2,"
    )

accum = int(
        input(
            "Select one of the following accumulator sizes: \n128 \n256 \n512 \n1024\n"
        )
    )
if not accum in [128, 256, 512, 1024]:
    raise ValueError(
        "'accum' parameter must be one of the following values: 128, 256, 512 or 1024."
    )

names = [f"dim{i}" for i in range(accum//8)]
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
    c_uc = "cleaned"
else:
    c_uc = "uncleaned"

csv_list = []
for i in tqdm(device_list):
    target_dir = f"{path_to_simhash}{i}/accum_{accum}/window_{window}/combo_{combo}/{c_uc}/"
    for j in os.listdir(target_dir):
        csv_list.append(target_dir + j)

name_of_current_data = f"FlexHash-PvLvC-accum_{accum}-window_{window}-combo_{combo}-cleaned"
# dataset = combine_csv_category(csv_list, names)
dataset = combine_csv(csv_list, names)
dataset.reset_index(drop=True, inplace=True)

print(f"*** Total samples in {name_of_current_data}: {len(dataset.index)} ***")
for device_name in sorted(dataset["class"].unique()):
    num_samples = len((dataset[dataset["class"] == device_name]).index)
    print(
        f"*** Samples for device: {device_name} in {name_of_current_data}: {num_samples} ({num_samples/dataset.shape[0]}%) ***"
    )

# x is the entire dataframe except for the class column
x = dataset.drop(["class"], axis=1)

# y_original is an unaltered list of all values in the class column
y_original = dataset["class"].values.tolist()

# # y is a dataframe of only the class column and the values have been converted to numeric representation
# counter = 0
# y_temp = dataset["class"].tolist()
# for unique_value in sorted(dataset["class"].unique()):
#     for index, value in enumerate(dataset["class"]):
#         if value == unique_value:
#             y_temp[index] = counter
#     counter += 1
# dataset["class"] = y_temp

y = dataset["class"]

x_train, x_test, y_train, y_test = train_test_split(
    x.values, y.values, test_size=0.2, stratify=y.values
)

names = list(range(x_train.shape[1]))
train_dataset_df = pd.DataFrame(x_train, columns=names)
train_dataset_df.insert(train_dataset_df.shape[1], "class", y_train)

query_string = "plug"
# print(train_dataset_df["class"]=="cam-3")
print(train_dataset_df["class"].str.startswith(query_string))
# print(train_dataset_df[train_dataset_df["class"]=="cam-3"])
print(train_dataset_df[train_dataset_df["class"].str.startswith(query_string)])
print(train_dataset_df[train_dataset_df["class"].str.startswith(query_string)].index)

# print(train_dataset_df[train_dataset_df["class"]])
exit(0)

# y is a dataframe of only the class column and the values have been converted to numeric representation
counter = 0
y_temp = dataset["class"].tolist()
for unique_value in sorted(dataset["class"].unique()):
    for index, value in enumerate(dataset["class"]):
        if value == unique_value:
            y_temp[index] = counter
    counter += 1
dataset["class"] = y_temp

# train_dataset_df[train_dataset_df["class"][:4] == "plug"] = "plug"
# train_dataset_df[train_dataset_df["class"][:5] == "light"] = "light"
# train_dataset_df[train_dataset_df["class"][:3] == "cam"] = "cam"

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
).fit(train_dataset_td, presets="best_quality")

results = predictor.fit_summary()

predictor = TabularPredictor.load(model_save_path)

all_devices_best_model_performance = [[
    "0_precision", "0_recall", "0_f1-score", "0_support",
    "1_precision", "1_recall", "1_f1-score", "1_support",
    "2_precision", "2_recall", "2_f1-score", "2_support",
    "accuracy",
    "macro_avg_precision", "macro_avg_recall", "macro_avg_f1-score", "macro_avg_support",
    "weighted_avg_precision", "weighted_avg_recall", "weighted_avg_f1-score", "weighted_avg_support"
]]

test_dataset_td = TabularDataset(test_dataset_df)
eval = predictor.evaluate(test_dataset_td, detailed_report=True, silent=False)
# print(eval["classification_report"].keys())
output_list = []
for key in eval["classification_report"].keys():
    if key in ["0", "1", "2", "macro avg", "weighted avg"]:
        for key2 in eval["classification_report"][key].keys():
            output_list.append(eval["classification_report"][key][key2])
    else:
        output_list.append(eval["classification_report"][key])
all_devices_best_model_performance.append(output_list)

# eval = predictor.evaluate(test_dataset_td[test_dataset_td["class"]==0], detailed_report=True, silent=False)
# best_model_output_list = [accum, window, combo, 0]
# for key in eval.keys():
#     best_model_output_list.append(eval[key])
# all_devices_best_model_performance.append(best_model_output_list)
#
# eval = predictor.evaluate(test_dataset_td[test_dataset_td["class"]==1], detailed_report=True, silent=False)
# best_model_output_list = [accum, window, combo, 1]
# for key in eval.keys():
#     best_model_output_list.append(eval[key])
# all_devices_best_model_performance.append(best_model_output_list)
#
# eval = predictor.evaluate(test_dataset_td[test_dataset_td["class"]==2], detailed_report=True, silent=False)
# best_model_output_list = [accum, window, combo, 2]
# for key in eval.keys():
#     best_model_output_list.append(eval[key])
# all_devices_best_model_performance.append(best_model_output_list)

output_df = pd.DataFrame(all_devices_best_model_performance)
output_df.to_csv(f"{accum}_{window}_{combo}_PvLvC_{gethostname()}.csv", index=False)
