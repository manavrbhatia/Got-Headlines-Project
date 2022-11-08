import argparse
import dataSort
import os


def main():
    parser = argparse.ArgumentParser(description="Automatic News Title Generation")
    parser.add_argument('exp_name', type=str, default="generic")
    arguments = parser.parse_args()

    exp_name = arguments.exp_name

    if exp_name == "generic":
        if os.path.exists("../data/generic-dataset.csv"):
            print("Dataset already found, skipping write.")
        else:
            print("Dataset not found, writing.")
            dataSort.select_dataset("../data/generic-dataset.csv")
            print("Wrote generic dataset to file")


if __name__ == "__main__":
    main()