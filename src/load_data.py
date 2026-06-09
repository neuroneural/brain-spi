""" FBIRN ICA dataset loading script"""

import numpy as np
import pandas as pd

def load_data(
    file_path: str = "/Users/ppopov1/_archive/2026 OHBM (SPIs and holo)/adm-proj/data/fbirn_ica/fbirn_data.npz",
    demo_path: str = "/Users/ppopov1/_archive/2026 OHBM (SPIs and holo)/adm-proj/data/fbirn_ica/demographics_legend.csv"
    ):
    """
    Loads FBIRN data saved in npz archive.

    Args:
        file_path (str): The path to the .npz file.
        demo_path (str): The path to the demographics legend CSV file.

    Returns:
        A dictionary containing the loaded arrays:
            - data: The main data array of shape (311, 140, 53) (samples, time points, features).
            - diags: Diagnosis labels.
            - sexes: Sex labels.    
            - ages: Age values.
            - age_bins: Age bin labels.
        A pandas DataFrame containing the demographics legend.
    """
    
    npz = np.load(file_path, allow_pickle=True)
    data = npz["data"]
    diags = npz["diags"]
    sexes = npz["sexes"]
    ages = npz["ages"]
    age_bins = npz["age_bins"]

    demo_df = pd.read_csv(demo_path)

    return {
        "data": data,
        "diags": diags,
        "sexes": sexes,
        "ages": ages,
        "age_bins": age_bins
    }, demo_df

if __name__ == "__main__":
    # Example of how to use the function to load the data
    data_dict, demo_df = load_data()

    # Print the keys of the loaded data to verify
    print("Arrays loaded from the file:", data_dict.keys())
    print("Data shape:", data_dict["data"].shape)
    print("Others shape:", data_dict["diags"].shape, data_dict["sexes"].shape, data_dict["ages"].shape, data_dict["age_bins"].shape)
    print("Demographics legend DataFrame:")
    print(demo_df)