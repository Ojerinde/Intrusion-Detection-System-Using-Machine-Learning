import os
import pandas as pd


def load_dataset(split="train", data_dir="data"):
    """
    Load the NSL-KDD dataset from local files.

    Parameters:
    - split (str): Either 'train' or 'test' to select the dataset split.
    - data_dir (str): Directory where the dataset files are located.

    Returns:
    - pd.DataFrame: Loaded dataset.
    """
    if split not in ("train", "test"):
        raise ValueError("split must be either 'train' or 'test'")

    file_path = os.path.join(data_dir, f"{split}.csv")

    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"{file_path} not found. Please ensure the dataset is in the '{data_dir}' directory."
        )

    return pd.read_csv(file_path)
