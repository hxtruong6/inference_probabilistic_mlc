"""Subset the NIH metadata CSV to the official test-list image IDs.

Reads NIH's test_list.txt (one image filename per line) and the cleaned
metadata CSV (produced by fix-csv.py); writes a CSV containing only the
metadata rows whose Image Index appears in test_list.txt.

Usage:
    python create_test_set.py <test_list.txt> <data_entry_fixed.csv> <output.csv>
"""
import argparse
import pandas as pd


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("test_list_txt", help="NIH-provided test_list.txt")
    ap.add_argument("data_entry_csv", help="Cleaned Data_Entry_2017 CSV (from fix-csv.py)")
    ap.add_argument("output_csv", help="Destination CSV for the test-set subset")
    args = ap.parse_args()

    with open(args.test_list_txt, "r") as f:
        test_filenames = set(line.strip() for line in f if line.strip())

    df = pd.read_csv(args.data_entry_csv)
    if "Unnamed: 11" in df.columns:
        df = df.drop(columns=["Unnamed: 11"])

    out = df[df["Image Index"].isin(test_filenames)]
    out.to_csv(args.output_csv, index=False)
    print(f"Wrote {len(out)} test rows to {args.output_csv}")


if __name__ == "__main__":
    main()
