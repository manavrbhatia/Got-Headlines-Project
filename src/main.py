import argparse
import dataSort
import os
from constants import INPUT_FILE

from datasets import load_dataset
from preprocessing import tokenize_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq

from model import generic_TD5_model

def main():
    parser = argparse.ArgumentParser(description="Automatic News Title Generation")
    parser.add_argument('exp_name', type=str, default="generic")
    parser.add_argument('--exp_name', type=str, default="generic")
    arguments = parser.parse_args()

    exp_name = arguments.exp_name
    dataset = None
    model_name = ""

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

    

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    print("Loaded Tokenizer.")

    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    print("Loaded Model.")

    split_tokenized_dataset = tokenize_dataset(dataset["train"], tokenizer)
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    trainer = None
    if exp_name == "generic": 
        trainer = generic_TD5_model(split_tokenized_dataset, 
        data_collator,
        model=model,
        tokenizer=tokenizer)
    
    trainer.train()
    


if __name__ == "__main__":
    main()