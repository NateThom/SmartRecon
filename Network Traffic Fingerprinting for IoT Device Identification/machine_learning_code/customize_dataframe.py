import pandas as pd
import os

# Get hash csv file paths
path_to_csvs = "../hashes_cleaned_confusionMatrix/"

csv_names = sorted(os.listdir(path_to_csvs), reverse=True)
csv_names_full = []
for csv_name in csv_names:
    csv_names_full.append(path_to_csvs + csv_name)

# Load dataset
columns = ['dim1', 'dim2', 'dim3', 'dim4', 'dim5', 'dim6', 'dim7', 'dim8', 'dim9', 'dim10', 'dim11',
           'dim12', 'dim13', 'dim14', 'dim15', 'dim16', 'dim17', 'dim18', 'dim19', 'dim20', 'dim21',
           'dim22', 'dim23', 'dim24', 'dim25', 'dim26', 'dim27', 'dim28', 'dim29', 'dim30', 'dim31', 'dim32', 'class']

for dataset_index, dataset_name in enumerate(csv_names_full):
    print(f"Dataset Name: {dataset_name}")
    dataset = pd.read_csv(dataset_name, names=columns)

    dataset = dataset[dataset["class"] != "Gosuna_LightBulb"]
    dataset = dataset[dataset["class"] != "Gosuna_Socket"]
    dataset = dataset[dataset["class"] != "Lumiman_Bulb600"]
    dataset = dataset[dataset["class"] != "Lumiman_Bulb900"]
    dataset = dataset[dataset["class"] != "Lumiman_SmartPlug"]
    dataset = dataset[dataset["class"] != "Ocean_Radio"]
    dataset = dataset[dataset["class"] != "Smart_Lamp"]
    dataset = dataset[dataset["class"] != "Smart_LightStrip"]
    dataset = dataset[dataset["class"] != "oossxx_SmartPlug"]
    dataset = dataset[dataset["class"] != "Renpho_SmartPlug"]

    dataset = dataset.tail(dataset.shape[0]-1)

    dataset.to_csv(dataset_name, index=False, header=False)