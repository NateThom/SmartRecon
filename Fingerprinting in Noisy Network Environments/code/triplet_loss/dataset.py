import pandas as pd
from torch.utils.data import Dataset
import os
import random
import torch

class Triplet_Dataset(Dataset):
    def __init__(self, args, fold):
        self.train = args.train
        self.test = args.test

        self.dataset_dir = args.dataset_dir
        self.dataset = args.dataset

        csv_filename = os.listdir(args.dataset_dir+args.dataset)[0]

        # Read the labels from the specified file
        self.data = pd.read_csv(args.dataset_dir+args.dataset+csv_filename, names=args.dataset_column_names)

        self.total_dataset_length = len(self.data.index)

        # y is a dataframe of only the class column and the values will be converted to numeric representation
        # Tokenize the values in y so that they have a numeric representation
        counter = 0
        y_temp = self.data['class'].tolist()
        class_label_dict = {}
        for unique_value in sorted(self.data['class'].unique()):
            class_label_dict[counter] = unique_value
            for index, value in enumerate(self.data['class']):
                if value == unique_value:
                    y_temp[index] = counter
            counter += 1
        self.data["class"] = y_temp
        labels_numeric = self.data['class'].unique()

        self.dataset = pd.DataFrame(columns=args.dataset_column_names)

        for device_name in labels_numeric:
            # Get the part of the dataset which pertains to the current device
            current_device_dataset = self.data[self.data['class'] == device_name]
            # Shuffle the part of the dataset for the current device
            current_device_dataset_length = len(current_device_dataset.index)

            if fold == "training":
                self.dataset = pd.concat(
                    [self.dataset, current_device_dataset[int(current_device_dataset_length * .2):]])
            elif fold == "validation":
                self.dataset = pd.concat(
                    [self.dataset, current_device_dataset[int(current_device_dataset_length * .1):]])
            elif fold == "testing":
                self.dataset = pd.concat(
                    [self.dataset, current_device_dataset[int(current_device_dataset_length * .1):int(current_device_dataset_length * .2)]])

        self.total_dataset_length = len(self.dataset.index)
        self.dataset_no_class = torch.from_numpy(self.dataset.drop('class', axis=1).values.astype('float32'))
        self.dataset_class = self.dataset['class'].values.tolist()

        self.dataset_by_class = {}
        for device_name in labels_numeric:
            entry = self.dataset[self.dataset['class'] == device_name]
            self.dataset_by_class[device_name] = [torch.from_numpy(entry.drop('class', axis=1).values.astype('float32')),
                                                  len(entry.index)-1]

    def __len__(self):
        return self.total_dataset_length

    def __getitem__(self, idx):
        # anchor = torch.from_numpy(self.dataset.iloc[idx].drop('class').values.astype('float32'))
        anchor = self.dataset_no_class[idx]
        anchor_class = self.dataset_class[idx]

        pos_index = random.randint(0, self.dataset_by_class[anchor_class][1])
        positive = self.dataset_by_class[anchor_class][0][pos_index]

        neg_class_choices = [choice for choice in self.dataset_by_class.keys() if choice not in [anchor_class]]
        neg_class_choice = random.choice(neg_class_choices)
        neg_index = random.randint(0, self.dataset_by_class[neg_class_choice][1])
        negative = self.dataset_by_class[neg_class_choice][0][neg_index]

        return anchor, anchor_class, positive, negative
