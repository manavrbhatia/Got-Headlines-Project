from transformers import Seq2SeqTrainer, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments
from constants import BATCH_SIZE, NUM_EPOCHS
from preprocessing import tokenize_dataset

def generic_TD5_model(tokenized_datasets, data_collator, model, tokenizer):

    logging_steps = len(tokenized_datasets["train"]) // BATCH_SIZE

    args = Seq2SeqTrainingArguments(
        output_dir=f"results/generic-results",
        evaluation_strategy="epoch",
        learning_rate=5.6e-5,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=NUM_EPOCHS,
        predict_with_generate=True,
        logging_steps=logging_steps
    )

    tokenized_datasets = tokenized_datasets.remove_columns(
        ["article", "title", "year", "publication"]
    )

    trainer = Seq2SeqTrainer(
        model,
        args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
        #compute_metrics=compute_metrics,

    return trainer