import argparse
import dataSort
import smallPubDataSort
import os
from constants import INPUT_FILE
import wandb

from datasets import load_dataset, load_from_disk
from preprocessing import tokenize_year_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq

from model import get_trainer_ally
import torch
import wandb


def main():

    # Testing this remove later

    parser = argparse.ArgumentParser(description="Automatic News Title Generation")
    parser.add_argument('exp_name', type=str, default="generic")
    #Remove if not dist
    parser.add_argument("--local_rank", type=int)
    parser.add_argument("--train_year", type=int)

    arguments = parser.parse_args()

    exp_name = arguments.exp_name
    train_year = arguments.train_year
    local_rank = arguments.local_rank

    wandb.init(project="EECS595 Final Project", entity="salamentic", group="Practical year pipeline for "+str(train_year))

    model_name = "google/pegasus-x-base"
    split_tokenized_dataset = None
    next_year_tokenized = None
    prev_year_tokenized = None

    tokenizer = AutoTokenizer.from_pretrained(model_name, max_length=1024, return_tensors='pt')
    print("Loaded Pegasus X Tokenizer")

    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    print("Loaded Model.")

    data_collator = DataCollatorForSeq2Seq()

    # Assume if one year present, all should be present
    if not os.path.exists(f"../data/year_data/tokenized-{train_year}"):
        for year in [2016,2017,2018,2019,2020]:
            if os.path.exists(f"../data/year_data/{year}-dataset.csv"):
                print("Dataset already found, skipping write.")
            else:
                dataSort.select_dataset("../data/generic-all-dataset.csv", begin_year=year, end_year=year)
                print("Wrote generic dataset to file")

            year_dataset = load_dataset(
                "csv",
                data_files="../data/generic-all-dataset.csv")
            if year == train_year:
                split_tokenized_dataset = tokenize_year_dataset(year_dataset["train"], tokenizer, year)
            elif year == train_year-1:
                next_year_tokenized = tokenize_year_dataset(year_dataset["train"], tokenizer, year)
            elif year == train_year+1:
                prev_year_tokenized = tokenize_year_dataset(year_dataset["train"], tokenizer, year)
        print("Wrote Tokenized datasets")

    elif os.path.exists("../data/tokenized-"+exp_name):
        print("Tokenized dataset found")
        split_tokenized_dataset = load_from_disk(f"../data/year_data/tokenized-{year}")
        next_year_tokenized  = load_from_disk(f"../data/year_data/tokenized-{year+1}") if train_year != 2020 else None
        prev_year_tokenized = load_from_disk(f"../data/year_data/tokenized-{year-1}") if train_year != 2016 else None


    trainer = get_trainer_ally(split_tokenized_dataset,
    data_collator,
    model=model,
    tokenizer=tokenizer,
    exp_name=exp_name,
    rank=local_rank)

    if train_year != 2016:
        print(f"Evaluating on {train_year-1} year gives us:", trainer.evaluate(prev_year_tokenized["test"]))
    trainer.train()
    print("Evaluating on this year gives us", trainer.evaluate())
    if train_year != 2020:
        print("Evaluating on next year gives us", trainer.evaluate(next_year_tokenized["test"])

    print("Finished Execution")

if __name__ == "__main__":
    main()
