import pandas as pd
from torch.utils.data import Dataset
import os
import random

class Triplet_Dataset(Dataset):
    def __init__(self, args, fold):
        if fold == "training":
            lower_bound = 0
            upper_bound = args.train_size
        elif fold == "validation":
            lower_bound = args.train_size
            upper_bound = args.train_size + args.val_size
        elif fold == "testing":
            lower_bound = args.train_size + args.val_size
            upper_bound = args.train_size + args.val_size + args.test_size
        elif fold == "not test":
            lower_bound = 0
            upper_bound = args.train_size + args.val_size


        self.train = args.train
        self.test = args.test

        self.dataset_dir = args.dataset_dir
        self.dataset = args.dataset

        csv_filename = os.listdir(args.dataset_dir+args.dataset)[0]

        # Read the labels from the specified file
        self.inputs = pd.read_csv(args.dataset_dir+args.dataset+csv_filename,
                                  names=args.dataset_column_names)

        self.total_dataset_size = len(self.inputs.index)
        self.inputs = self.inputs[int(self.total_dataset_size*lower_bound):int(self.total_dataset_size*upper_bound)]

        # print the total number of samples in each data set and the number of samples per device in each data set
        if args.print_num_samples_bool:
            print(f"*** Parameters in {args.dataset}: {len(self.inputs.index)} ***")
            for device_name in self.inputs["class"].unique():
                num_samples = len((self.inputs[self.inputs["class"] == device_name]).index)
                print(
                    f"*** Samples for device: {device_name} in {args.dataset}: {num_samples} ({num_samples / self.inputs.shape[0]}) ***")

        # y is a dataframe of only the class column and the values will be converted to numeric representation
        # Tokenize the values in y so that they have a numeric representation
        counter = 0
        y_temp = self.inputs['class'].tolist()
        class_label_dict = {}
        for unique_value in sorted(self.inputs['class'].unique()):
            class_label_dict[counter] = unique_value
            for index, value in enumerate(self.inputs['class']):
                if value == unique_value:
                    y_temp[index] = counter
            counter += 1
        self.inputs["class"] = y_temp
        labels_numeric = self.inputs['class'].unique()

        dataset_shuffled = self.inputs.sample(frac=1)
        temp_device




























































































































        "?"

        ';'
        for device_name in labels_numeric:
            # Get the part of the dataset which pertains to the current device
            dataset_current_device = dataset_shuffled[dataset_shuffled['class'] == device_name]
            # Shuffle the part of the dataset for the current device
            dataset_current_device_length = len(dataset_current_device.index)

            self.x[device_name] = {}
            self.y[device_name] = {}
            self.x[device_name]["test"] = []
            self.y[device_name]["test"] = []

            temp_shuffled_test = dataset_current_device[:int(dataset_current_device_length * .2)]
            temp_shuffled_train = dataset_current_device[int(dataset_current_device_length * .2):]

            self.x[device_name]["test"] = (
                        self.x[device_name]["test"] + temp_shuffled_test.drop(['class'], axis=1).values.tolist())
            self.y[device_name]["test"] = (self.y[device_name]["test"] + temp_shuffled_test['class'].values.tolist())

            self.x["train"] = (self.x["train"] + temp_shuffled_train.drop(['class'], axis=1).values.tolist())
            self.y["train"] = (self.y["train"] + temp_shuffled_train['class'].values.tolist())

            temp2 = list(zip(self.x["train"], self.y["train"]))
            random.shuffle(temp2)
            self.x["train"], self.y["train"] = [[i for i, j in temp2], [j for i, j in temp2]]

    def __len__(self):
        return self.total_dataset_size

    def __getitem__(self, idx):
        anchor = 0

        positive = 0

        negative = 0

        return(anchor, positive, negative)