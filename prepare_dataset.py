import os
import shutil
import pandas as pd
csv_path = "archive/train_labels.csv"
source_folder = "archive/train"
dest_with_mask = "training/dataset/with_mask"
dest_without_mask = "training/dataset/without_mask"
os.makedirs(dest_with_mask, exist_ok=True)
os.makedirs(dest_without_mask, exist_ok=True)
df = pd.read_csv(csv_path)
for index, row in df.iterrows():
    filename = row["filename"]
    label = row["label"]
    source_path = os.path.join(source_folder, filename)
    if label == "with_mask":
        shutil.copy(source_path, dest_with_mask)
    else:
        shutil.copy(source_path, dest_without_mask)
print("Dataset separation completed successfully!")