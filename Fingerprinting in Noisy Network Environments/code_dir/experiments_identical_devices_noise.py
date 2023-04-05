import gc
import os
from socket import gethostname

import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from autogluon.tabular import TabularDataset, TabularPredictor

from utils import combine_csv

path_to_simhash = "/home/nthom/Documents/SmartRecon/Fingerprinting in Noisy Network Environments/data/simhashes/"
path_to_noise = "/home/nthom/Documents/SmartRecon/Fingerprinting in Noisy Network Environments/data/FlexHash_noise/"

device = int(
            input("Select one of the following: \n1. Plug \n2. Light \n3. Cam\n")
        )
if not device in [1, 2, 3]:
    raise ValueError(
        "'device' parameter must be one of the following values: 1, 2 or 3. 1 represents plugs, 2 "
        "represents light bulbs and 3 represents cameras."
    )
if device == 1:
    device = "plug"
elif device == 2:
    device = "light"
elif device == 3:
    device = "cam"

# noise = int(
#             input("Select one of the following: \n1. Random \n2. IoT \n3. Network\n")
#         )
# if not noise in [1, 2, 3]:
#     raise ValueError(
#         "'noise' parameter must be one of the following values: 1, 2 or 3. 1 represents random noise, 2 "
#         "represents IoT noise and 3 represents network noise."
#     )
# if noise == 1:
#     noise = "random_noise"
# elif noise == 2:
#     noise = "iot_noise"
# elif noise == 3:
#     noise = "network_noise"

c_uc = int(input("Select one of the following: \n1. Cleaned \n2. Uncleaned\n"))
if not c_uc in [1, 2]:
    raise ValueError(
        "'c_uc' selection must be one of the following values: 1 or 2,"
    )
if c_uc == 1:
    c_uc = "cleaned"
else:
    c_uc = "uncleaned"

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

csv_list = []

target_dir = f"{path_to_simhash}{device}/accum_{accum}/window_{window}/combo_{combo}/{c_uc}/"
for j in os.listdir(target_dir):
    csv_list.append(target_dir + j)
csv_list.append(
    f"{path_to_noise}iot_noise/accum_{accum}/window_{window}/combo_{combo}/"
    f"iot_noise_iot_noise_per_packet_{c_uc}_{accum}_{window}_{combo}.csv"
)
csv_list.append(
    f"{path_to_noise}network_noise/accum_{accum}/window_{window}/combo_{combo}/"
    f"network_noise_network_noise_per_packet_{c_uc}_{accum}_{window}_{combo}.csv"
)

name_of_current_data = f"FlexHash-identicalDevice-noise-{device}-accum_{accum}-window_{window}-combo_{combo}-{c_uc}"
dataset = combine_csv(csv_list, names)
dataset.reset_index(drop=True, inplace=True)

print(f"*** Total samples in {name_of_current_data}: {len(dataset.index)} ***")
total_non_noise_device_count = 0
for device_name in sorted(dataset["class"].unique()):
    num_samples = len((dataset[dataset["class"] == device_name]).index)
    if device_name not in ["iot_noise", "network_noise"]:
        total_non_noise_device_count += num_samples
    print(
        f"*** Samples for device: {device_name} in {name_of_current_data}: {num_samples} ({num_samples/dataset.shape[0]}%) ***"
    )

filtered_dataset = dataset.loc[dataset['class'] != 'iot_noise']
filtered_dataset = filtered_dataset.loc[dataset['class'] != 'network_noise']
n_iot_noise_dataset = dataset.loc[dataset['class'] == 'iot_noise'][:total_non_noise_device_count]
n_network_noise_dataset = dataset.loc[dataset['class'] == 'network_noise'][:total_non_noise_device_count]
dataset = pd.DataFrame.append(filtered_dataset, n_iot_noise_dataset, n_network_noise_dataset)

print(f"*** Total samples in {name_of_current_data}: {len(dataset.index)} ***")
for device_name in sorted(dataset["class"].unique()):
    num_samples = len((dataset[dataset["class"] == device_name]).index)
    print(
        f"*** Samples for device: {device_name} in {name_of_current_data}: {num_samples} ({num_samples/dataset.shape[0]}%) ***"
    )

# x is the entire dataframe except for the class column
x = dataset.drop(["class"], axis=1)

y = dataset["class"]

x_train, x_test, y_train, y_test = train_test_split(
    x.values, y.values, test_size=0.2, stratify=y.values, random_state=0
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
    # y_original,
    x_train,
    y_train,
    x_test,
    y_test,
    names,
    dataset,
    # y_temp
)

model_save_path = f"agModels-{name_of_current_data}_{gethostname()}"

train_dataset_td = TabularDataset(train_dataset_df)
label = "class"
print("Summary of class variable: \n", train_dataset_td[label].describe())

predictor = TabularPredictor(
    eval_metric="accuracy", label="class", path=model_save_path
).fit(
    train_dataset_td,
    presets="best_quality",
    excluded_model_types=["CAT", "KNN", "RF", "FASTAI", "LR", "NN_TORCH", "AG_AUTOMM"],
    )
results = predictor.fit_summary()

#####
test_dataset_metric_df = pd.DataFrame()
unique_classes = sorted(test_dataset_df["class"].unique())

predictor = TabularPredictor.load(model_save_path)
test_dataset_td = TabularDataset(test_dataset_df)
test_dataset_predictions = predictor.predict_multi(test_dataset_td, models=None)
print(test_dataset_predictions)

accuracy_list = []
f1_list = []
precision_list = []
recall_list = []
model_list = []
device_list = []
for model in tqdm(test_dataset_predictions.keys()):
    for value in unique_classes:
        device_list.append(value)
        model_list.append(model)
    device_list.append("all")
    model_list.append("all")

    test_dataset_matrix = confusion_matrix(
        y_true=test_dataset_df["class"].values,
        y_pred=test_dataset_predictions[model],
        labels=unique_classes,
    )

    # test_dataset_metric_df["Accuracy"] = test_dataset_matrix.diagonal()/test_dataset_matrix.sum(axis=1)
    # print(test_dataset_metric_df)
    average_accuracy = 0
    for value in test_dataset_matrix.diagonal()/test_dataset_matrix.sum(axis=1):
        accuracy_list.append(value)
        average_accuracy += value
    average_accuracy /= 8
    accuracy_list.append(average_accuracy)

    test_dataset_f1 = f1_score(
        y_true=test_dataset_df["class"].values,
        y_pred=test_dataset_predictions[model],
        labels=unique_classes,
        average=None
    )
    # print(test_dataset_f1)

    # test_dataset_metric_df["F1"] = test_dataset_f1
    # print(test_dataset_metric_df)
    average_f1 = 0
    for value in test_dataset_f1:
        f1_list.append(value)
        average_f1 += value
    average_f1 /= 8
    f1_list.append(average_f1)

    test_dataset_precision = precision_score(
        y_true=test_dataset_df["class"].values,
        y_pred=test_dataset_predictions[model],
        labels=unique_classes,
        average=None
    )
    # print(test_dataset_precision)

    # test_dataset_metric_df["Precision"] = test_dataset_precision
    # print(test_dataset_metric_df)
    average_precision = 0
    for value in test_dataset_precision:
        precision_list.append(value)
        average_precision += value
    average_precision /=8
    precision_list.append(average_precision)

    test_dataset_recall = recall_score(
        y_true=test_dataset_df["class"].values,
        y_pred=test_dataset_predictions[model],
        labels=unique_classes,
        average=None
    )
    # print(test_dataset_recall)

    # test_dataset_metric_df["Recall"] = test_dataset_recall
    # print(test_dataset_metric_df)
    average_recall = 0
    for value in test_dataset_recall:
        recall_list.append(value)
        average_recall += value
    average_recall /=8
    recall_list.append(average_recall)

# print(model
test_dataset_metric_df["Model"] = model_list
test_dataset_metric_df["Accuracy"] = accuracy_list
test_dataset_metric_df["F1"] = f1_list
test_dataset_metric_df["Precision"] = precision_list
test_dataset_metric_df["Recall"] = recall_list
test_dataset_metric_df["Device"] = device_list

test_dataset_metric_df.to_csv(
    f"{device}_{accum}_{window}_{combo}_identicalDevices_noise_{gethostname()}.csv", index=False
)

