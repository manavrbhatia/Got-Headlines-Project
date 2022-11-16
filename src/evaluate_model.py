import argparse

from datasets import load_dataset
from preprocessing import tokenize_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq
import torch

def main():

    # Testing this remove later

    parser = argparse.ArgumentParser(description="Automatic News Title Generation Evaluation")
    parser.add_argument('exp_name', type=str, default="generic")
    parser.add_argument('article', type=str, default="generic")


    arguments = parser.parse_args()
    exp_name = arguments.exp_name
    input_file = arguments.article

    model_name = ""
    if exp_name == "generic_allyears":
        model_name = "google/mt5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    print("Loaded Tokenizer.")

    model = None
    if exp_name == "generic_allyears":
        model = AutoModelForSeq2SeqLM.from_pretrained("results/generic-results/checkpoint-5718")
    print("Loaded Model.")

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    input_art = ""
    with open(input_file) as f:
        input_art = f.read()
    encoded_art = tokenizer.encode(input_art, return_tensors='pt')
    pred = model.generate(encoded_art, max_length=30,do_sample=True, early_stopping=True)
    print(pred)
    decoded_preds = tokenizer.decode(pred[0], skip_special_tokens=True)

    model.save_pretrained("results/models/generic-results-ally")
    print("The title generated is:")
    print(decoded_preds)
    print("Finished Execution")

if __name__ == "__main__":
    main()
