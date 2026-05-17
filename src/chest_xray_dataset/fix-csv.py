"""Clean the `Patient Age` column in the NIH metadata CSV.

NIH's Data_Entry_2017.csv stores Patient Age as strings like "058Y";
this script strips the trailing "Y" and casts to int.

Usage:
    python fix-csv.py <input.csv> <output.csv>
"""
import argparse
import pandas as pd


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input_csv", help="Raw NIH Data_Entry_2017.csv")
    ap.add_argument("output_csv", help="Destination for cleaned CSV")
    args = ap.parse_args()

    df = pd.read_csv(args.input_csv)
    df["Patient Age"] = df["Patient Age"].str.replace("Y", "").astype(int)
    df.to_csv(args.output_csv, index=False)
    print(f"Wrote {len(df)} rows to {args.output_csv}")


if __name__ == "__main__":
    main()
