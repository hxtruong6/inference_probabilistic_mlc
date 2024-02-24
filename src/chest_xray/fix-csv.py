import pandas as pd


df = pd.read_csv(
    "/Users/xuantruong/Documents/JAIST/inference_prob_mlc_code/datasets/NIH/Data_Entry_2017.csv"
)

print(df.info())
print(df.head())
print(df["Patient Age"].value_counts())
# "058Y" -> "58"

df["Patient Age"] = df["Patient Age"].str.replace("Y", "").astype(int)

df.to_csv(
    "/Users/xuantruong/Documents/JAIST/inference_prob_mlc_code/datasets/NIH/Data_Entry_2017_fixed.csv",
    index=False,
)
