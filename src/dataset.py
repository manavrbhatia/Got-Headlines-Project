MAX_ARTICLE_LEN = 512
MAX_TITLE_LEN = 512

def preprocess_dataset(examples):
    model_inputs = tokenizer(
        examples["article"],
        max_length=MAX_ARTICLE_LEN,
        truncation=True,
    )

    labels = tokenizer(
        examples["title"], max_length=MAX_ARTICLE_LEN,
        truncation=True,
    )

    model_inputs["labels"] = labels["token_ids"]
    model_inputs["labels_mask"] = labels["attention_mask"]
    return model_inputs

def tokenize_dataset(dataset):
    train_test_data =  dataset.train_test_split(0.8)
    test_valid_data = train_test_data["test"].train_test_split(0.5)

    original_datasets = DatasetDict({
        "train": train_test_data["train"],
        "test": test_valid_data["train"],
        "valid": test_valid_data["test"],
    }
    )

    tokenized_datasets = original_datasets.map(preprocess_dataset, batched=True)