MAX_ARTICLE_LEN = 512

# Check distribution of lengths``
MAX_TITLE_LEN = 100

from datasets import DatasetDict

def tokenize_dataset(dataset, tokenizer, test_split=0.2):
    '''
    Take an untokenized, unsplit dataset, split it into 3
    '''
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

    return tokenized_datasets