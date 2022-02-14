import pandas as pd
from pandas import read_csv
import random
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold
import seaborn as sns
import matplotlib.pyplot  as plt
import wandb
import os
import numpy as np
import pickle

def setup_menu():
    # Get data set selection
    dataset_list = [["../hashes_cleaned", "Cleaned Hashes"], ["hashes_uncleaned", "Uncleaned Hashes"],
                    ["../hashes_cleaned_noRing/", "Cleaned Hashes No Ring"],
                    ["../hashes_cleaned_noRing_noTpBulb/", "Cleaned Hashes No TP Bulb"]]
    print("Select data set to run:")
    print("Options:")
    for index, dataset_name in enumerate(dataset_list):
        print(f"{index + 1}: {dataset_name}")

    input_bool = False
    while input_bool:
        dataset_index = int(input("Selection: "))
        if dataset_index >= 1 and dataset_index <= len(dataset_list):
            input_bool = True
        else:
            print("Invalid Selection.")

    # Get print num samples selection
    print("__________")
    print("Print number of samples per device? (Enter either 0 or 1)")
    input_bool = False
    while input_bool:
        print_num_samples_bool = int(input("Selection: "))
        if print_num_samples_bool >= 0 and print_num_samples_bool <= 1:
            input_bool = True
        else:
            print("Invalid Selection.")

        # Get prototyping selection
        print("__________")
        print("Use only a small percentage of the data set for prototyping? (Enter either 0 or 1")
        input_bool = False
        while input_bool:
            prototyping_bool = int(input("Selection: "))
            if prototyping_bool >= 0 and prototyping_bool <= 1:
                input_bool = True
            else:
                print("Invalid Selection.")

    path_to_csvs = dataset_list[dataset_index][0]
    name_of_current_data = dataset_list[dataset_index][1]
    print_num_samples_bool = bool(print_num_samples_bool)
    prototyping_bool = bool(prototyping_bool)

    return path_to_csvs, name_of_current_data, print_num_samples_bool, prototyping_bool

wandb.init(project="project_name", entity="your_entity_name")

# Get hash csv file paths
path_to_csvs, name_of_current_data, print_num_samples_bool, prototyping_bool = setup_menu()

csv_names = sorted(os.listdir(path_to_csvs), reverse=True)
csv_names_full = []
for csv_name in csv_names:
    csv_names_full.append(path_to_csvs + csv_name)

# Columns in the data set csv files
columns = ['dim1', 'dim2', 'dim3', 'dim4', 'dim5', 'dim6', 'dim7', 'dim8', 'dim9', 'dim10', 'dim11',
           'dim12', 'dim13', 'dim14', 'dim15', 'dim16', 'dim17', 'dim18', 'dim19', 'dim20', 'dim21',
           'dim22', 'dim23', 'dim24', 'dim25', 'dim26', 'dim27', 'dim28', 'dim29', 'dim30', 'dim31', 'dim32', 'class']

dataset_count = 10
dataset_dict = {}
metrics_list = []
for dataset_index, dataset_name in enumerate(csv_names_full):
    print(f"*** Begin Processing {dataset_name} Dataset ***")

    dataset_dict[dataset_name] = dataset_count
    dataset_count -= 1

    # Read the data set into RAM from memory
    dataset = read_csv(dataset_name, names=columns)

    # Truncate the data set for prototyping the code (i.e. quicker run time)
    if prototyping_bool:
        dataset = dataset.head(len(dataset.index) // 10)

    # print the total number of samples in each data set and the number of samples per device in each data set
    if print_num_samples_bool:
        print(f"*** Parameters in {dataset_name}: {dataset.shape[0]} ***")
        for device_name in dataset["class"].unique():
            num_samples = len((dataset[dataset["class"] == device_name]).index)
            print(f"*** Samples for device: {device_name} in {dataset_name}: {num_samples} ({num_samples/dataset.shape[0]}%) ***")

    # x is the entire dataframe except for the class column
    x = dataset.drop(['class'], axis=1)

    # y_original is an unaltered list of all values in the class column
    y_original = dataset['class'].values.tolist()

    # y is a dataframe of only the class column and the values will be converted to numeric representation
    # Tokenize the values in y so that they have a numeric representation
    y = dataset['class']
    counter = 0
    y_temp = dataset['class'].tolist()

    for unique_value in sorted(y.unique()):
        for index, value in enumerate(y):
            if value == unique_value:
                y_temp[index] = counter
        counter += 1

    dataset["class"] = y_temp
    y = dataset['class']
    labels_numeric = dataset['class'].unique()

    print("*** Dataset Loaded ***")

    print("*** Begin Generating Cross Folds ***")
    # This part of the code is for saving the random splits used for each crossfold
    # This is important for reproducibility
    if not os.path.isdir(f"./saved_data/dataframes/{name_of_current_data}"):
        os.mkdir(f"./saved_data/dataframes/{name_of_current_data}")
    if not os.path.isdir(f"./saved_data/dataframes/{name_of_current_data}/{dataset_dict[dataset_name]}"):
        os.mkdir(f"./saved_data/dataframes/{name_of_current_data}/{dataset_dict[dataset_name]}")

    # Create datasets for 5 fold cross validation
    x = {0:{"train":[], "test":[]}, 1:{"train":[], "test":[]}, 2:{"train":[], "test":[]},
         3:{"train":[], "test":[]}, 4:{"train":[], "test":[]}, 5:{"train":[], "test":[]}}
    y = {0:{"train":[], "test":[]}, 1:{"train":[], "test":[]}, 2:{"train":[], "test":[]},
         3:{"train":[], "test":[]}, 4:{"train":[], "test":[]}, 5:{"train":[], "test":[]}}

    for device_name in labels_numeric:
        # Get the part of the dataset which pertains to the current device
        temp = dataset[dataset['class'] == device_name]
        # Shuffle the part of the dataset for the current device
        temp_shuffled = temp.sample(frac=1)
        # Pickle the dataframe for future use
        temp_shuffled.to_pickle(f"./saved_data/dataframes/{name_of_current_data}/{dataset_dict[dataset_name]}/"
                                f"shuffled_{name_of_current_data}-{dataset_dict[dataset_name]}_device-{device_name}_dataframe.pickle")
        length_temp_shuffled = len(temp_shuffled.index)
        for current_fold in range(5):
            if current_fold == 0:
                # Get 20% for use in testing
                temp_shuffled_test = temp_shuffled[:int(length_temp_shuffled * .2)]
                # Get 80% for use in training
                temp_shuffled_train = temp_shuffled[int(length_temp_shuffled * .2):]
            elif current_fold == 1:
                temp_shuffled_test = temp_shuffled[int(length_temp_shuffled * .2):int(length_temp_shuffled * .4)]
                dataframes = [temp_shuffled[:int(length_temp_shuffled * .2)], temp_shuffled[int(length_temp_shuffled * .4):]]
                temp_shuffled_train = pd.concat(dataframes, ignore_index=True)
            elif current_fold == 2:
                temp_shuffled_test = temp_shuffled[int(length_temp_shuffled * .4):int(length_temp_shuffled * .6)]
                dataframes = [temp_shuffled[:int(length_temp_shuffled * .2)], temp_shuffled[int(length_temp_shuffled * .4):]]
                temp_shuffled_train = pd.concat(dataframes, ignore_index=True)
            elif current_fold == 3:
                temp_shuffled_test = temp_shuffled[int(length_temp_shuffled * .6):int(length_temp_shuffled * .8)]
                dataframes = [temp_shuffled[:int(length_temp_shuffled * .6)], temp_shuffled[int(length_temp_shuffled * .8):]]
                temp_shuffled_train = pd.concat(dataframes, ignore_index=True)
            else:
                temp_shuffled_test = temp_shuffled[int(length_temp_shuffled * .8):]
                temp_shuffled_train = temp_shuffled[:int(length_temp_shuffled * .8)]

            x[current_fold]["test"] = (x[current_fold]["test"] + temp_shuffled_test.drop(['class'], axis=1).values.tolist())
            y[current_fold]["test"] = (y[current_fold]["test"] + temp_shuffled_test['class'].values.tolist())
            x[current_fold]["train"] = (x[current_fold]["train"] + temp_shuffled_train.drop(['class'], axis=1).values.tolist())
            y[current_fold]["train"] = (y[current_fold]["train"] + temp_shuffled_train['class'].values.tolist())

            # Randomly shuffle the resulting training dataset so that all samples for the same class are not passed in together
            temp2 = list(zip(x[current_fold]["train"], y[current_fold]["train"]))
            random.shuffle(temp2)
            x[current_fold]["train"], y[current_fold]["train"] = [[ i for i, j in temp2], [ j for i, j in temp2]]
            # x[current_fold]["train"], y[current_fold]["train"] = zip(*temp)
    print("*** Finished Generating Cross Folds ***")

    # Spot Check Algorithms
    models = []
    models.append((0, MLPClassifier()))
    models.append((1, MLPClassifier()))
    models.append((2, MLPClassifier()))
    models.append((3, MLPClassifier()))
    models.append((4, MLPClassifier()))
    print(models[0][1].get_params())

    # Create directories for saving the models. Important for reproducibility
    if not os.path.isdir(f"./saved_data/models/{name_of_current_data}"):
        os.mkdir(f"./saved_data/models/{name_of_current_data}")
    if not os.path.isdir(f"./saved_data/models/{name_of_current_data}/{dataset_dict[dataset_name]}"):
        os.mkdir(f"./saved_data/models/{name_of_current_data}/{dataset_dict[dataset_name]}")

    accuracy_dict = {"base": 0}
    precision_dict = {"base": 0}
    recall_dict = {"base": 0}
    f1_dict = {"base": 0}

    # evaluate each model
    for model_name, model in models:
        print(f"*** Begin Training {model_name} ***")
        print(len(y[model_name]["train"]))
        model.fit(x[model_name]["train"], y[model_name]["train"])

        filename = f"./saved_data/models/{name_of_current_data}/{dataset_dict[dataset_name]}/{model_name}.pickle"
        pickle.dump(model, open(filename, 'wb'))
        print(f"*** {model_name} Trained ***")

        # print(f"*** Calculate Predictions and Probabilities ***")
        y_pred = model.predict(x[model_name]["test"])
        y_probas = model.predict_proba(x[model_name]["test"])
        # print(f"*** Predictions and Probabilities Done ***")

        accuracy_dict[f"fold {model_name}"] = accuracy_score(y[model_name]["test"], y_pred)
        accuracy_dict["base"] += accuracy_score(y[model_name]["test"], y_pred)
        precision_dict[f"fold {model_name}"] = precision_score(y[model_name]["test"], y_pred, average='weighted')
        precision_dict[f"base"] += precision_score(y[model_name]["test"], y_pred, average='weighted')
        recall_dict[f"fold {model_name}"] = recall_score(y[model_name]["test"], y_pred, average='weighted')
        recall_dict["base"] += recall_score(y[model_name]["test"], y_pred, average='weighted')
        f1_dict[f"fold {model_name}"] = f1_score(y[model_name]["test"], y_pred, average='weighted')
        f1_dict["base"] += f1_score(y[model_name]["test"], y_pred, average='weighted')

        wandb.log({f"Fold {model_name} accuracy on {name_of_current_data}": accuracy_dict[f"fold {model_name}"],
                   "Dataset": dataset_dict[dataset_name],
                   "Num Samples": dataset.shape[0]})
        wandb.log({f"Fold {model_name} precision on {name_of_current_data}": precision_dict[f"fold {model_name}"],
                   "Dataset": dataset_dict[dataset_name],
                   "Num Samples": dataset.shape[0]})
        wandb.log({f"Fold {model_name} recall on {name_of_current_data}": recall_dict[f"fold {model_name}"],
                   "Dataset": dataset_dict[dataset_name],
                   "Num Samples": dataset.shape[0]})
        wandb.log({f"Fold {model_name} f1 on {name_of_current_data}": f1_dict[f"fold {model_name}"],
                   "Dataset": dataset_dict[dataset_name],
                   "Num Samples": dataset.shape[0]})

    # print("*** Begin Metric Plotting ***")

    accuracy_dict["base"] /= len(models)
    precision_dict[f"base"] /= len(models)
    recall_dict["base"] /= len(models)
    f1_dict["base"] /= len(models)

    wandb.log({f"Total accuracy on {name_of_current_data}": accuracy_dict["base"],
               "Dataset": dataset_dict[dataset_name],
               "Num Samples": dataset.shape[0]})
    wandb.log({f"Total precision on {name_of_current_data}": precision_dict["base"],
               "Dataset": dataset_dict[dataset_name],
               "Num Samples": dataset.shape[0]})
    wandb.log({f"Total recall on {name_of_current_data}": recall_dict["base"],
               "Dataset": dataset_dict[dataset_name],
               "Num Samples": dataset.shape[0]})
    wandb.log({f"Total f1 on {name_of_current_data}": f1_dict["base"],
               "Dataset": dataset_dict[dataset_name],
               "Num Samples": dataset.shape[0]})
