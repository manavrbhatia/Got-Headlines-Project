from transformers import Seq2SeqTrainer, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments
from constants import BATCH_SIZE, NUM_EPOCHS
from preprocessing import tokenize_dataset
import numpy as np
import evaluate


def compute_metrics(pred):
    rouge_score = evaluate.load("rouge")
    predictions, labels = pred
    # Decode generated summaries into text
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

    # Decode reference summaries into text
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # ROUGE expects a newline after each sentence
    decoded_preds = ["\n".join(sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(sent_tokenize(label.strip())) for label in decoded_labels]
    # Compute ROUGE scores
    result = rouge_score.compute(
        predictions=decoded_preds, references=decoded_labels, use_stemmer=True
    )

    # Extract the median scores
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
    return {k: round(v, 4) for k, v in result.items()}

def generic_TD5_model(tokenized_datasets, data_collator, model, tokenizer, evaluator):

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
        eval_dataset=tokenized_datasets["valid"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    return trainer