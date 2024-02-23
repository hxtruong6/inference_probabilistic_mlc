def get_image_indices_from_txt(txt_file_path, csv_file_path):
    """
    Extracts Image Index values from the given CSV file, filtering
    to include only those found in the specified TXT file.

    Args:
        txt_file_path (str): Path to the TXT file containing image filenames.
        csv_file_path (str): Path to the CSV file containing image metadata.

    Returns:
        list: A list of Image Index values present in the TXT file.
    """

    # 1. Read image filenames from TXT file
    with open(txt_file_path, "r") as f:
        image_filenames = f.read().splitlines()

    # 2. Extract Image Index from filenames (efficiently)
    image_indices_from_txt = [filename for filename in image_filenames]

    print(image_indices_from_txt[:5])
    # 3. Load CSV and filter Image Index values
    import pandas as pd

    df = pd.read_csv(csv_file_path)
    print(df.head())
    df.drop(columns=["Unnamed: 11"], inplace=True)

    filtered_image_indices = df[df["Image Index"].isin(image_indices_from_txt)]

    return filtered_image_indices


# Example Usage
txt_file_path = "/Users/xuantruong/Documents/JAIST/inference_prob_mlc_code/datasets/NIH/test_list.txt"
csv_file_path = "/Users/xuantruong/Documents/JAIST/inference_prob_mlc_code/datasets/NIH/Data_Entry_2017.csv"

matching_image_indices = get_image_indices_from_txt(txt_file_path, csv_file_path)
print(matching_image_indices.head())

# Save to CSV
matching_image_indices.to_csv(
    "/Users/xuantruong/Documents/JAIST/inference_prob_mlc_code/datasets/NIH/Data_Entry_2017__testset.csv",
    index=False,
)
