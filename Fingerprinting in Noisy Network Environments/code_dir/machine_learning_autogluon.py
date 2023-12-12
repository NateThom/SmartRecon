import gc
from socket import gethostname

import pandas as pd
from sklearn.model_selection import train_test_split
from autogluon.tabular import TabularDataset, TabularPredictor

import utils

dataset, name_of_current_data = utils.get_dataset()
dataset.reset_index(drop=True, inplace=True)
collected = gc.collect()
print("Garbage collector: collected %d objects." % (collected))

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

# y is a dataframe of only the class column and the values have been converted to numeric representation
y = dataset["class"]
counter = 0
y_temp = dataset["class"].tolist()
for unique_value in sorted(y.unique()):
    for index, value in enumerate(y):
        if value == unique_value:
            y_temp[index] = counter
    counter += 1
dataset["class"] = y_temp
y = dataset["class"]
labels_numeric = dataset["class"].unique()

x_train, x_test, y_train, y_test = train_test_split(
    x.values, y.values, test_size=0.8, stratify=y.values
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
    y_temp,
    labels_numeric,
    x_train,
    y_train,
    x_test,
    y_test,
    dataset,
    names,
)
collected = gc.collect()
print("Garbage collector: collected %d objects." % (collected))

print("*** Dataset Loaded ***")

model_save_path = f"agModels-{name_of_current_data}_{gethostname()}"

train_dataset_td = TabularDataset(train_dataset_df)
label = "class"
print("Summary of class variable: \n", train_dataset_td[label].describe())

predictor = TabularPredictor(
    eval_metric="f1_micro", label="class", path=model_save_path
).fit(train_dataset_td, presets="best_quality")

results = predictor.fit_summary()

predictor = TabularPredictor.load(model_save_path)

test_dataset_td = TabularDataset(test_dataset_df)
y_test = test_dataset_td[label]
test_data_noLabel = test_dataset_td.drop(columns=[label])

y_pred = predictor.predict(test_data_noLabel)
perf = predictor.evaluate_predictions(
    y_true=y_test, y_pred=y_pred, auxiliary_metrics=True
)

leaderboard_df = predictor.leaderboard(test_dataset_td, silent=True)
leaderboard_df.to_csv(
    f"autogluon_leaderboard_{name_of_current_data}_{gethostname()}.csv"
)
