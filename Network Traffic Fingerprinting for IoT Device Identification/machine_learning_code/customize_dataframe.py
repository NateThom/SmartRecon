import pandas as pd
import os

def remove_class(class_name, dataset):
    dataset = dataset[dataset["class"] != class_name]
    return dataset

# Remove list of classes from a group of datasets

# path_to_csvs = "../hashes_cleaned_confusionMatrix/"
#
# csv_names = sorted(os.listdir(path_to_csvs), reverse=True)
# csv_names_full = []
# for csv_name in csv_names:
#     csv_names_full.append(path_to_csvs + csv_name)
#
# # Load dataset
# columns = ['dim1', 'dim2', 'dim3', 'dim4', 'dim5', 'dim6', 'dim7', 'dim8', 'dim9', 'dim10', 'dim11',
#            'dim12', 'dim13', 'dim14', 'dim15', 'dim16', 'dim17', 'dim18', 'dim19', 'dim20', 'dim21',
#            'dim22', 'dim23', 'dim24', 'dim25', 'dim26', 'dim27', 'dim28', 'dim29', 'dim30', 'dim31', 'dim32', 'class']
#
# classes_to_remove = ["Gosuna_LightBulb", "Gosuna_Socket", "Lumiman_Bulb600", "Lumiman_Bulb900", "Lumiman_SmartPlug",
#                      "Ocean_Radio", "Smart_Lamp", "Smart_LightStrip", "oossxx_SmartPlug", "Renpho_SmartPlug"
#                      ]
#
# # for dataset_index, dataset_name in enumerate(csv_names_full):
# #     print(f"Dataset Name: {dataset_name}")
# #     dataset = pd.read_csv(dataset_name, names=columns)
# #     for item in classes_to_remove:
# #         dataset = remove_class(item, dataset)

names = ['dim1','dim2','dim3','dim4','dim5','dim6','dim7','dim8','dim9','dim10','dim11',
         'dim12','dim13','dim14','dim15','dim16','dim17','dim18','dim19','dim20','dim21',
         'dim22','dim23','dim24','dim25','dim26','dim27','dim28','dim29','dim30','dim31','dim32','class']

path_to_datasets = "/home/nthom/Documents/hashdata/new_data/same_plug_uncleaned/"
device_csvs = os.listdir(path_to_datasets)
if "combined.csv" in device_csvs:
    device_csvs.remove("combined.csv")

# final_csv = pd.read_csv(path_to_datasets + device_csvs[0])
final_df = pd.DataFrame(columns=names)

for index, csv in enumerate(device_csvs):
    temp_df = pd.read_csv(path_to_datasets + csv, names=names)

    final_df = pd.concat([final_df, temp_df])

final_df.to_csv(path_to_datasets + "combined.csv", index=False)