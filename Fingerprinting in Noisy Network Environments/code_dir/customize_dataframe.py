import pandas as pd
import os

def remove_class(class_name, dataset):
    dataset = dataset[dataset["class"] != class_name]
    return dataset

names = ['dim1','dim2','dim3','dim4','dim5','dim6','dim7','dim8','dim9','dim10','dim11',
         'dim12','dim13','dim14','dim15','dim16','dim17','dim18','dim19','dim20','dim21',
         'dim22','dim23','dim24','dim25','dim26','dim27','dim28','dim29','dim30','dim31','dim32','class']

# path_to_csv = "/home/nthom/Documents/SmartRecon/Fingerprinting in Noisy Network Environments/data/batyr_uncleaned/per-packet-hashes-uncleaned.csv"
# df_to_categorize = pd.read_csv(path_to_csv, names=names)
# unique_classes1 = df_to_categorize["class"].unique()
#
# df_to_categorize.replace(to_replace="Tenvis_Cam", value="Camera", inplace=True)
# df_to_categorize.replace(to_replace="Wans_Cam", value="Camera", inplace=True)
# df_to_categorize.replace(to_replace="Ring_Doorbell", value="Camera", inplace=True)
# df_to_categorize.replace(to_replace="Chime_Doorbell", value="Camera", inplace=True)
# df_to_categorize.replace(to_replace="D-Link_Cam936L", value="Camera", inplace=True)
# df_to_categorize.replace(to_replace="itTiot_Cam", value="Camera", inplace=True)
#
# df_to_categorize.replace(to_replace="Minger_LightStrip", value="Light", inplace=True)
# df_to_categorize.replace(to_replace="Lumiman_Bulb900", value="Light", inplace=True)
# df_to_categorize.replace(to_replace="Lumiman_Bulb600", value="Light", inplace=True)
# df_to_categorize.replace(to_replace="Smart_Lamp", value="Light", inplace=True)
# df_to_categorize.replace(to_replace="Gosuna_LightBulb", value="Light", inplace=True)
# df_to_categorize.replace(to_replace="Smart_LightStrip", value="Light", inplace=True)
# df_to_categorize.replace(to_replace="tp-link_LightBulb", value="Light", inplace=True)
#
# df_to_categorize.replace(to_replace="tp-link_SmartPlug", value="Plug", inplace=True)
# df_to_categorize.replace(to_replace="Renpho_SmartPlug", value="Plug", inplace=True)
# df_to_categorize.replace(to_replace="oossxx_SmartPlug", value="Plug", inplace=True)
# df_to_categorize.replace(to_replace="Gosuna_Socket", value="Plug", inplace=True)
# df_to_categorize.replace(to_replace="Wemo_SmartPlug", value="Plug", inplace=True)
# df_to_categorize.replace(to_replace="Lumiman_SmartPlug", value="Plug", inplace=True)
#
# df_to_categorize.replace(to_replace="LaCrosse_AlarmClock", value="Misc", inplace=True)
# df_to_categorize.replace(to_replace="Goumia_Coffemaker", value="Misc", inplace=True)
# df_to_categorize.replace(to_replace="Ocean_Radio", value="Misc", inplace=True)
#
# unique_classes2 = df_to_categorize["class"].unique()
#
# df_to_categorize.to_csv("/home/nthom/Documents/SmartRecon/Fingerprinting in Noisy Network Environments/data/batyr_uncleaned_categories/uncleaned-categories.csv", index=False)

# Remove list of classes from a group of datasets
path_to_datasets = "/home/nthom/Documents/SmartRecon/Fingerprinting in Noisy Network Environments/data/clddr_categories_cleaned_network_noise_uncleaned/"
device_csvs = os.listdir(path_to_datasets)
if "combined.csv" in device_csvs:
    device_csvs.remove("combined.csv")

final_df = pd.DataFrame(columns=names)

for index, csv in enumerate(device_csvs):
    temp_df = pd.read_csv(path_to_datasets + csv, names=names)

    final_df = pd.concat([final_df, temp_df])

# final_df.to_csv(path_to_datasets + "combined.csv", index=False)
