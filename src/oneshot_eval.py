import argparse

from datasets import load_dataset
from preprocessing import tokenize_dataset_fewshot
from model import get_trainer_fewshot
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq
import torch

publications = ["Axios", "Business Insider", "Buzzfeed News", "CNBC", "CNN", "Economist",
"Fox News", "Gizmodo", "Hyperallergic", "Mashable", "New Republic", "New Yorker", "People",
"Politico", "Refinery 29", "Reuters", "TMZ", "TechCrunch", "The Hill", "The New York Times",
"The Verge", "Vice", "Vice News", "Vox", "Washington Post", "Wired"]

def main():
    parser = argparse.ArgumentParser(description="Automatic News Title Generation Evaluation")
    parser.add_argument('exp_name', type=str, default="generic")

    arguments = parser.parse_args()
    exp_name = arguments.exp_name

    model_name = ""
    if exp_name == "generic_allyears":
        model_name = "google/mt5-small"
    elif exp_name == "pegx_ally":
        model_name = "google/pegasus-X-base"

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    print("Loaded Tokenizer.")

    for publication in publications:
        if os.path.exists("../data/"+publication+".csv"):
            print("Dataset already found, skipping write.")
        else:
            smallPubDataSort.generate()
            print("Wrote generic dataset to file")
        temp_dataset = load_dataset(
            "csv",
            data_files="../data/"+publication+".csv")
        datasets.append(tokenize_dataset_fewshot(temp_dataset, tokenizer, publication, 0.9))


    model = None
    if exp_name == "generic_allyears":
        model = AutoModelForSeq2SeqLM.from_pretrained("results/generic-results/checkpoint-5718")
    else:
        print("Experiment TBD")
        return

    print("Loaded Model.")

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    for pub_i, pub_dataset in enumerate(datasets):
        model = None
        if exp_name == "generic_allyears":
            model = AutoModelForSeq2SeqLM.from_pretrained("results/generic-results/checkpoint-5718")
        pub_trainer = get_trainer_fewshot(pub_dataset, data_collator, model, tokenizer, publications[pub_i])
        print("Doing Fewshot for "+pub_name)
        pub_trainer.train()
        pub_trainer.evaluate()
        print("Finished Fewshot for "+pub_name)

    print("Finished Execution")

if __name__ == "__main__":
    main()
