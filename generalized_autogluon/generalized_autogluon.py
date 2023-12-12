import yaml
import argparse
import os
from time import time

import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from autogluon.tabular import TabularDataset, TabularPredictor

def parse_args():
    parser = argparse.ArgumentParser(
            description='A PyTorch project for face recognition.')
    parser.add_argument('--config', 
            help='train config file path')
    args = parser.parse_args()
    return args

def combine_csv(csv_list, names):
    final_df = pd.DataFrame(columns=names)
    temp_df_list = []
    for csv in tqdm(csv_list):
        # temp_df = pd.read_csv(csv, names=names, skiprows=1)
        temp_df_list.append(pd.read_csv(csv, names=names, skiprows=1))
        # final_df = pd.concat([final_df, temp_df])
    final_df = pd.concat(temp_df_list)

    return final_df

def plot_confusion_matrix(confusion_matrix_df, save_path):
    row_sums = confusion_matrix_df.sum(axis=1)
    confusion_matrix_df = confusion_matrix_df.div(row_sums, axis=1)*100
    
    # Create a heatmap for the confusion matrix
    plt.figure(figsize=(10, 10))
    plt.rcParams.update({'font.size': 14})
    ax = sns.heatmap(confusion_matrix_df, annot=True, fmt='2.2f', cmap="crest", cbar=False)
    for t in ax.texts: t.set_text(t.get_text() + " %")
    plt.xticks(rotation=40)
    plt.yticks(rotation=40)
    plt.xlabel('Predicted Labels', fontsize=20)
    plt.ylabel('True Labels', fontsize=20)

    # Save the plot to EPS format
    plt.savefig(save_path+".pdf")
    # plt.savefig(save_path+".png", format='png')
    plt.close()

def main():
    args = parse_args()

    with open(args.config, 'r') as f:
            config = yaml.load(f, yaml.SafeLoader)

    if not os.path.isdir(f"./{config['name']}"):
        os.mkdir(f"./{config['name']}")
    if not os.path.isdir(f"./{config['name']}/results/"):
        os.mkdir(f"./{config['name']}/results/")                    
    if not os.path.isdir(f"./{config['name']}/models/"):
        os.mkdir(f"./{config['name']}/models/")

    for accumulator in config["data"]["accumulator"]:
        for window in config["data"]["window"]:
            for combo in range(2, window+1):
                print(accumulator, window, combo)
                # Check to_skip config before running
                if f"{accumulator}_{window}_{combo}" in config["data"]["to_skip"]:
                    print(f"Skipping: {accumulator}_{window}_{combo}")
                    continue
                start_time = time()
                if config["downsample"]["sample_ratio"] == "SelectMaxN":
                    if not os.path.isdir(f"./{config['name']}/results/{accumulator}_{window}_{combo}_{config['model_quality']}_stratify-{config['downsample']['stratify_by']}-{config['downsample']['sample_ratio']}-{config['downsample']['MaxN']}"):
                        os.mkdir(f"./{config['name']}/results/{accumulator}_{window}_{combo}_{config['model_quality']}_stratify-{config['downsample']['stratify_by']}-{config['downsample']['sample_ratio']}-{config['downsample']['MaxN']}")
                else:
                    if not os.path.isdir(f"./{config['name']}/results/{accumulator}_{window}_{combo}_{config['model_quality']}_stratify-{config['downsample']['stratify_by']}-{config['downsample']['sample_ratio']}"):
                        os.mkdir(f"./{config['name']}/results/{accumulator}_{window}_{combo}_{config['model_quality']}_stratify-{config['downsample']['stratify_by']}-{config['downsample']['sample_ratio']}")

                csv_list = []
                label_list = []
                csv_column_names_list = []
                for class_index in range(len(config['data']["classes"])):
                    class_data = config['data']["classes"][class_index]["class"]
                    label = class_data["label"]
                    outer = class_data['path_to_outer_data_dir']
                    inner = class_data['path_to_inner_data_dir']
                    device = class_data['device']

                    csv_column_names = [f"dim{i}" for i in range(accumulator//8)]
                    csv_column_names.append("class")

                    csv_path = outer + inner + f"{device}_{accumulator}_{window}_{combo}.csv"
                    
                    csv_list.append(csv_path)
                    label_list.append(label)
                    csv_column_names_list.append(csv_column_names)

                first_list_len = len(csv_column_names_list[0])
                for sublist in csv_column_names_list[1:]:
                    if len(sublist) != first_list_len:
                        print("Error! Not all of the classes defined in the configuration have the same accumulator length.")
                        exit(0)

                dataset = combine_csv(csv_list, csv_column_names_list[0])
                dataset.reset_index(drop=True, inplace=True)
                for query_string in label_list:
                    dataset.loc[dataset[dataset["class"].str.startswith(query_string)].index, "class"] = query_string

                print(f"*** Total samples in {config['name']}_{accumulator}_{window}_{combo}: {len(dataset.index)} ***")
                for device_name in sorted(dataset["class"].unique()):
                    num_samples = len((dataset[dataset["class"] == device_name]).index)
                    print(
                        f"*** Samples for device: {device_name} in {config['name']}_{accumulator}_{window}_{combo}: {num_samples} ({num_samples/dataset.shape[0]}%) ***"
                    )

                #######
                # Define the stratification feature
                stratify_by = config["downsample"]["stratify_by"]
                # Define the sampling ratio
                sampling_ratio = config["downsample"]["sample_ratio"]

                if sampling_ratio == "minimum":
                    min_count = float("inf")
                    for unique_class in dataset["class"].unique():
                        unique_class_sample_count = dataset[dataset["class"] == unique_class].shape[0]
                        if unique_class_sample_count < min_count:
                            min_count = unique_class_sample_count
                    sample_sizes = (dataset[stratify_by].value_counts().apply(lambda x: min(x, min_count))).astype(int)
                elif sampling_ratio == "SelectMaxN":
                    MaxN = config["downsample"]["MaxN"]
                    # Calculate the number of samples to take from each category
                    sample_sizes = (dataset[stratify_by].value_counts().apply(lambda count: min(MaxN, count)))
                else:
                    # Calculate the number of samples to take from each category
                    sample_sizes = (dataset[stratify_by].value_counts() * sampling_ratio).astype(int)

                # Perform stratified sampling
                dataset = dataset.groupby(stratify_by, group_keys=False).apply(lambda x: x.sample(sample_sizes.loc[x.name]))
                # Reset the index if desired
                dataset.reset_index(drop=True, inplace=True)
                #######

                print(f"*** Total samples in {config['name']}_{accumulator}_{window}_{combo}: {len(dataset.index)} ***")
                for device_name in sorted(dataset["class"].unique()):
                    num_samples = len((dataset[dataset["class"] == device_name]).index)
                    print(
                        f"*** Samples for device: {device_name} in {config['name']}_{accumulator}_{window}_{combo}: {num_samples} ({num_samples/dataset.shape[0]}%) ***"
                    )

                # x is the entire dataframe except for the class column
                x = dataset.drop(["class"], axis=1)

                # y_original is an unaltered list of all values in the class column
                y_original = dataset["class"].values.tolist()

                y = dataset["class"]

                if config["mode"] == "LoadAndTest":
                    test_dataset_df = pd.read_csv(f"./{config['load_root']}/results/{accumulator}_{window}_{combo}_{config['load_model_quality']}_stratify-{config['load_downsample']['stratify_by']}-{config['load_downsample']['sample_ratio']}/test_data.csv")
                    test_dataset_td = TabularDataset(test_dataset_df)
                    label = "class"

                    print("\n##########\n")
                    print("Summary of class variable in test dataset: \n", test_dataset_td[label].describe())
                    print("\n##########\n")

                else:
                    x_train, x_test, y_train, y_test = train_test_split(
                        x.values, y.values, test_size=.2, stratify=y.values
                    )

                    names = list(range(x_train.shape[1]))
                    train_dataset_df = pd.DataFrame(x_train, columns=names)
                    train_dataset_df.insert(train_dataset_df.shape[1], "class", y_train)

                    names = list(range(x_test.shape[1]))
                    test_dataset_df = pd.DataFrame(x_test, columns=names)
                    test_dataset_df.insert(test_dataset_df.shape[1], "class", y_test)

                    # cams
                    # device_nameshj = train_dataset_df["class"].str.split("_").str[0]
                    # device_numbers = train_dataset_df["class"].str.split("-").str[-1]
                    # train_dataset_df["class"] = device_names + '-' + device_numbers
                    # device_names = test_dataset_df["class"].str.split("_").str[0]
                    # device_numbers = test_dataset_df["class"].str.split("-").str[-1]
                    # test_dataset_df["class"] = device_names + '-' + device_numbers

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
                    )

                    train_dataset_td = TabularDataset(train_dataset_df)
                    test_dataset_td = TabularDataset(test_dataset_df)
                    label = "class"
                    
                    print("\n##########\n")
                    print("Summary of class variable in train dataset: \n", train_dataset_td[label].describe())
                    print("\n##########\n")
                    print("Summary of class variable in test dataset: \n", test_dataset_td[label].describe())
                    print("\n##########\n")

                if config["downsample"]["sample_ratio"] == "SelectMaxN":
                    model_save_path = f"./{config['name']}/models/{accumulator}_{window}_{combo}_{config['model_quality']}_stratify-{config['downsample']['stratify_by']}-{config['downsample']['sample_ratio']}-{config['downsample']['MaxN']}"
                else:
                    model_save_path = f"./{config['name']}/models/{accumulator}_{window}_{combo}_{config['model_quality']}_stratify-{config['downsample']['stratify_by']}-{config['downsample']['sample_ratio']}"

                if config["mode"] == "LoadAndTest":
                    load_root = config["load_root"]
                    
                    if config["downsample"]["sample_ratio"] == "SelectMaxN":
                        load_path = f"./{load_root}/models/{accumulator}_{window}_{combo}_{config['load_model_quality']}_stratify-{config['load_downsample']['stratify_by']}-{config['load_downsample']['sample_ratio']}-{config['downsample']['MaxN']}"
                    else:
                        load_path = f"./{load_root}/models/{accumulator}_{window}_{combo}_{config['load_model_quality']}_stratify-{config['load_downsample']['stratify_by']}-{config['load_downsample']['sample_ratio']}"
                    predictor = TabularPredictor.load(load_path)
                else:
                    predictor = TabularPredictor(
                        eval_metric="accuracy", label="class", path=model_save_path
                    ).fit(
                        train_dataset_td,
                        presets=config["model_quality"],
                        excluded_model_types=["CAT", "KNN", "RF", "FASTAI", "LR", "NN_TORCH", "AG_AUTOMM"],
                    )

                output_list = []
                output_list_column_names = []

                runtime = time() - start_time
                output_list_column_names.append(f"runtime")
                output_list.append(runtime)

                # test_dataset_td = TabularDataset(test_dataset_df)
                eval = predictor.evaluate(test_dataset_td, detailed_report=True, silent=True)
                    
                for col in eval["confusion_matrix"].columns:
                    true_positives = eval["confusion_matrix"].loc[col, col]
                    # total_samples = eval["confusion_matrix"].loc[:, col].sum()
                    total_samples = eval["confusion_matrix"].loc[col, :].sum()
                    accuracy = true_positives/total_samples
                    eval["classification_report"][col]["accuracy"] = accuracy

                if config["downsample"]["sample_ratio"] == "SelectMaxN":
                    plot_confusion_matrix(eval["confusion_matrix"], f"./{config['name']}/results/{accumulator}_{window}_{combo}_{config['model_quality']}_stratify-{config['downsample']['stratify_by']}-{config['downsample']['sample_ratio']}-{config['downsample']['MaxN']}/confusionMatrix")
                else:
                    plot_confusion_matrix(eval["confusion_matrix"], f"./{config['name']}/results/{accumulator}_{window}_{combo}_{config['model_quality']}_stratify-{config['downsample']['stratify_by']}-{config['downsample']['sample_ratio']}/confusionMatrix")

                for key1 in eval:
                    if key1 == "confusion_matrix":
                        continue
                    elif key1 == "classification_report":
                        for key2 in eval[key1]:
                            if key2 == "accuracy":
                                continue
                            else:
                                for key3 in eval[key1][key2]:
                                    output_list_column_names.append(f"{key2}-{key3}")
                                    output_list.append(eval[key1][key2][key3])
                    else:
                        output_list_column_names.append(key1)
                        output_list.append(eval[key1])

                output_df = pd.DataFrame(np.array(output_list).reshape((1, -1)), columns=output_list_column_names)
                if config["downsample"]["sample_ratio"] == "SelectMaxN":
                    output_df.to_csv(f"./{config['name']}/results/{accumulator}_{window}_{combo}_{config['model_quality']}_stratify-{config['downsample']['stratify_by']}-{config['downsample']['sample_ratio']}-{config['downsample']['MaxN']}/metrics.csv", index=False)
                    train_dataset_df.to_csv(f"./{config['name']}/results/{accumulator}_{window}_{combo}_{config['model_quality']}_stratify-{config['downsample']['stratify_by']}-{config['downsample']['sample_ratio']}-{config['downsample']['MaxN']}/train_data.csv", index=False)
                    test_dataset_df.to_csv(f"./{config['name']}/results/{accumulator}_{window}_{combo}_{config['model_quality']}_stratify-{config['downsample']['stratify_by']}-{config['downsample']['sample_ratio']}-{config['downsample']['MaxN']}/test_data.csv", index=False)
                else:
                    output_df.to_csv(f"./{config['name']}/results/{accumulator}_{window}_{combo}_{config['model_quality']}_stratify-{config['downsample']['stratify_by']}-{config['downsample']['sample_ratio']}/metrics.csv", index=False)
                    train_dataset_df.to_csv(f"./{config['name']}/results/{accumulator}_{window}_{combo}_{config['model_quality']}_stratify-{config['downsample']['stratify_by']}-{config['downsample']['sample_ratio']}/train_data.csv", index=False)
                    test_dataset_df.to_csv(f"./{config['name']}/results/{accumulator}_{window}_{combo}_{config['model_quality']}_stratify-{config['downsample']['stratify_by']}-{config['downsample']['sample_ratio']}/test_data.csv", index=False)

if __name__=="__main__":
    main()
