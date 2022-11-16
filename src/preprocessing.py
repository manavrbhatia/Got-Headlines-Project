MAX_ARTICLE_LEN = 512

# Check distribution of lengths``
MAX_TITLE_LEN = 100

from datasets import DatasetDict, load_from_disk
import os

def tokenize_dataset(dataset, tokenizer, experiment_name, test_split=0.2):
    '''
    Take an untokenized, unsplit dataset, split it into 3
    '''

    if os.path.exists("../data/tokenized-"+experiment_name):
        print("Tokenized dataset found")
        return load_from_disk("../data/tokenized-"+experiment_name)

    train_test_data =  dataset.train_test_split(test_split)
    test_valid_data = train_test_data["test"].train_test_split(0.5)

    original_datasets = DatasetDict({
        "train": train_test_data["train"],
        "test": test_valid_data["train"],
        "valid": test_valid_data["test"],
    }
    )

    def preprocess_dataset(examples):

        model_inputs = tokenizer(
            [str(x) for x in examples["article"]],
            max_length=MAX_ARTICLE_LEN,
            truncation=True,
        )

        labels = tokenizer(
            [str(x) for x in examples["title"]],
            max_length=MAX_TITLE_LEN,
            truncation=True,
        )

        model_inputs["labels"] = labels["input_ids"]
        model_inputs["labels_mask"] = labels["attention_mask"]
        return model_inputs

    tokenized_datasets = original_datasets.map(preprocess_dataset,
     batched=True,)
    tokenized_datasets.save_to_disk("../data/tokenized-"+experiment_name)

    return tokenized_datasets


def tokenize_dataset_fewshot(dataset, tokenizer, pub_name, test_split=0.9):
    '''
    Take an untokenized, unsplit dataset, split it into 3
    '''

    if os.path.exists("../data/tokenized-pubs/tokenized-fewshot-"+pub_name):
        print("Tokenized dataset found")
        return load_from_disk("../data/tokenized-pubs/tokenized-fewshot-"+pub_name)

    train_test_data =  dataset.train_test_split(test_split)

    def preprocess_dataset(examples):

        model_inputs = tokenizer(
            [str(x) for x in examples["article"]],
            max_length=MAX_ARTICLE_LEN,
            truncation=True,
        )

        print(examples["article"][0])

        labels = tokenizer(
            [str(x) for x in examples["title"]],
            max_length=MAX_TITLE_LEN,
            truncation=True,
        )

        model_inputs["labels"] = labels["input_ids"]
        model_inputs["labels_mask"] = labels["attention_mask"]
        return model_inputs

    tokenized_datasets = train_test_data.map(preprocess_dataset,
     batched=True)
    tokenized_datasets.save_to_disk("../data/tokenized-pubs/tokenized-fewshot-"+pub_name)

    return tokenized_datasets

def tokenize_year_dataset(dataset, tokenizer, year, test_split=0.2):
    '''
    Take an untokenized, unsplit dataset, split it into 3
    '''

    if os.path.exists("../data/year_data/tokenized-"+str(year)):
        print("Tokenized dataset found")
        return load_from_disk("../data/year_data/tokenized-"+str(year))

    train_test_data =  dataset.train_test_split(test_split)
    test_valid_data = train_test_data["test"].train_test_split(0.5)

    original_datasets = DatasetDict({
        "train": train_test_data["train"],
        "test": test_valid_data["train"],
        "valid": test_valid_data["test"],
    }
    )

    def preprocess_dataset(examples):

        model_inputs = tokenizer(
            [str(x) for x in examples["article"]],
            max_length=MAX_ARTICLE_LEN,
            truncation=True,
        )

        labels = tokenizer(
            [str(x) for x in examples["title"]],
            max_length=MAX_TITLE_LEN,
            truncation=True,
        )

        model_inputs["labels"] = labels["input_ids"]
        model_inputs["labels_mask"] = labels["attention_mask"]
        return model_inputs

    tokenized_datasets = original_datasets.map(preprocess_dataset,
     batched=True,)
    tokenized_datasets.save_to_disk("../data/year_data/tokenized-"+str(year))

    return tokenized_datasets

