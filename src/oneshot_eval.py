import argparse
import os
import torch

from datasets import load_dataset,load_from_disk
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq

from preprocessing import tokenize_dataset_fewshot
from model import get_trainer_fewshot
import smallPubDataSort


publications = ["Axios", "Business Insider", "Buzzfeed News", "CNBC", "Economist",
"Fox News", "Gizmodo", "Hyperallergic", "Mashable", "New Republic", "New Yorker", "People",
"Politico", "Refinery 29",  "TMZ", "TechCrunch", "The Hill",
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

    pub_datasets = []
    for publication in publications:
        if os.path.exists("../data/tokenized-pubs/tokenized-fewshot-"+publication):
            print("Tokenized dataset found for "+publication)
            pub_datasets.append(load_from_disk("../data/tokenized-pubs/tokenized-fewshot-"+publication))
        else:
            if os.path.exists("../data/"+publication+".csv"):
                print("Dataset already found, skipping write.")
            else:
                smallPubDataSort.generate()
                print("Wrote generic dataset to file")
            temp_dataset = load_dataset(
                "csv",
                data_files="../data/"+publication+".csv")
            pub_datasets.append(tokenize_dataset_fewshot(temp_dataset["train"], tokenizer, publication, 0.9))
            temp_dataset.cleanup_cache_files()

    model = None
    if exp_name == "generic_allyears":
        model = AutoModelForSeq2SeqLM.from_pretrained("results/models/generic-results-ally")
    else:
        print("Experiment TBD")
        return

    print("Loaded Model.")

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    for pub_i, pub_dataset in enumerate(pub_datasets):
        pub_trainer = get_trainer_fewshot(pub_dataset, data_collator, model, tokenizer, publications[pub_i])
        print("Doing initial evaluation for "+publications[pub_i])
        #print("")
        print("Evaluation is ", pub_trainer.evaluate())
        #print("")
        #pub_trainer.train()
        #print("")
        #print("Evaluation after fewshot is ", pub_trainer.evaluate())
        print("Finished Fewshot for "+publications[pub_i])
        print("")

        del model
        del pub_trainer
        torch.cuda.empty_cache()

        if exp_name == "generic_allyears":
            model = AutoModelForSeq2SeqLM.from_pretrained("results/models/generic-results-ally")

    print("Finished Execution")

if __name__ == "__main__":
    main()
