import argparse
import dataSort
import smallPubDataSort
import os
from constants import INPUT_FILE
import wandb

from datasets import load_dataset
from preprocessing import tokenize_dataset
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
    parser.add_argument("--publication", type=str)

    arguments = parser.parse_args()

    exp_name = arguments.exp_name

    publication = arguments.publication

    local_rank = arguments.local_rank
    wandb.init(project="EECS595 Final Project", entity="salamentic", group="Experiment: "+exp_name)


    dataset = None
    model_name = ""
    tokenizer = None

    if exp_name == "generic":
        if os.path.exists("../data/generic-dataset.csv"):
            print("Dataset already found, skipping write.")
        else:
            dataSort.select_dataset("../data/generic-dataset.csv")
            print("Wrote generic dataset to file")

        dataset = load_dataset(
            "csv",
            data_files="../data/generic-dataset.csv",
        )
        model_name = "google/mt5-small"

    if exp_name == "generic_allyears":
        if os.path.exists("../data/generic-all-dataset.csv"):
            print("Dataset already found, skipping write.")
        else:
            dataSort.select_dataset("../data/generic-all-dataset.csv", begin_year=2016, end_year=2020)
            print("Wrote generic dataset to file")

        dataset = load_dataset(
            "csv",
            data_files="../data/generic-all-dataset.csv",
        )
        model_name = "google/mt5-small"

    if exp_name == "smallPub":
        if os.path.exists("../data/"+publication+".csv"):
            print("Dataset already found, skipping write.")
        else:
            smallPubDataSort.generate()
            print("Wrote generic dataset to file")

        dataset = load_dataset(
            "csv",
            data_files="../data/"+publication+".csv",
        )
        model_name = "google/mt5-small"

    if exp_name == "pegx_ally":
        if os.path.exists("../data/generic-all-dataset.csv"):
            print("Dataset already found, skipping write.")
        else:
            dataSort.select_dataset("../data/generic-all-dataset.csv", begin_year=2016, end_year=2020)
            print("Wrote generic dataset to file")

        dataset = load_dataset(
            "csv",
            data_files="../data/generic-all-dataset.csv",
        )
        model_name = "google/pegasus-x-base"

    if exp_name == "pegx_ally":
        tokenizer = AutoTokenizer.from_pretrained(model_name, max_length=4096, return_tensors='pt')
        print("Loaded Pegasus X Tokenizer")
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        print("Loaded Tokenizer.")

    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    print("Loaded Model.")

    split_tokenized_dataset = tokenize_dataset(dataset["train"], tokenizer, exp_name)
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    trainer = None
    if exp_name == "generic" or exp_name == "generic_allyears" or exp_name == "pegx_ally":
        trainer = get_trainer_ally(split_tokenized_dataset,
        data_collator,
        model=model,
        tokenizer=tokenizer,
        exp_name=exp_name)

    trainer.train()
    trainer.evaluate()
    print("Finished Execution")

if __name__ == "__main__":
    main()
