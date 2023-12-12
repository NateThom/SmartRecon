import pandas as pd

path_to_output = "/storage/nate/SmartRecon/FlexHash/similar_devices_Hashes/cam_allwinner_cleaned_nilsimsa/"

# Load the CSV file
csv_file_path = '/storage/nate/SmartRecon/FlexHash/nilsimsa_cam_hashes/nilsimsa_cams.csv'
data = pd.read_csv(csv_file_path)

# Get the unique classes from the label column
unique_classes = data.iloc[:, -1].unique()

# Create a dictionary to hold the dataframes for each class
class_dataframes = {}

# Separate data into dataframes based on unique classes
for class_name in unique_classes:
    class_df = data[data.iloc[:, -1] == class_name]
    class_dataframes[class_name] = class_df

# Save each dataframe as a separate CSV file
for class_name, class_df in class_dataframes.items():
    class_csv_file = path_to_output + f'{class_name[:-2]}_allwinner{class_name[-2:]}_256_5_3.csv'
    class_df.to_csv(class_csv_file, index=False)
    print(f"Saved {class_name} data to {class_csv_file}")
