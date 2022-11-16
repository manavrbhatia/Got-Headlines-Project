from transformers import Seq2SeqTrainer, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments
from constants import BATCH_SIZE, NUM_EPOCHS
from preprocessing import tokenize_dataset
import numpy as np
import evaluate

from nltk import sent_tokenize

def compute_metrics(pred,tokenizer):
    rouge_score = evaluate.load("rouge")
    predictions, labels = pred
    # Decode generated summaries into text
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

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
    result = {key: value*100 for key, value in result.items()}
    return {k: round(v, 4) for k, v in result.items()}

def get_trainer_ally(tokenized_datasets, data_collator, model, tokenizer, exp_name, rank):

    logging_steps = len(tokenized_datasets["train"]) // (4*BATCH_SIZE*4)

    tokenize_compute = lambda x : compute_metrics(x, tokenizer)

    # Source for the deepspeed config template is https://www.kaggle.com/code/tanulsingh077/longformer-training-with-deepspeed-and-hf-trainer/notebook

    print("results/"+exp_name)
    args = Seq2SeqTrainingArguments(
        seed=42,
        output_dir=f"/scratch/eecs595f22_class_root/eecs595f22_class/anrao/results/"+exp_name,
        evaluation_strategy='epoch',
        learning_rate=8e-4,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=int(BATCH_SIZE/2),
        weight_decay=0.01,
        save_total_limit=1,
        num_train_epochs=NUM_EPOCHS,
        predict_with_generate=True,
        logging_strategy="epoch",
        logging_first_step=True,
        save_strategy="epoch",
        bf16=True,
        # Maybe change this back, idk if works
        bf16_full_eval=True,
        optim='adafactor',
        gradient_accumulation_steps=4,
        eval_accumulation_steps=4,
        #optim='adamw_torch',
        deepspeed="ds_config.json",
        generation_max_length=100,
        local_rank=rank,
        report_to="wandb",
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
        compute_metrics=tokenize_compute,
    )

    return trainer


def get_trainer_fewshot(tokenized_datasets, data_collator, model, tokenizer, pub_name):

    logging_steps = len(tokenized_datasets["train"]) // BATCH_SIZE

    tokenize_compute = lambda x : compute_metrics(x, tokenizer)

    # Source for the deepspeed config template is https://www.kaggle.com/code/tanulsingh077/longformer-training-with-deepspeed-and-hf-trainer/notebook

    args = Seq2SeqTrainingArguments(
        output_dir=f"fewshot_results/generic_ally/generic_ally_"+pub_name,
        evaluation_strategy="no",
        learning_rate=1e-6,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        save_total_limit=0,
        num_train_epochs=10,
        predict_with_generate=True,
        logging_strategy="no",
        save_strategy="no",
        fp16=True,
        lr_scheduler_type='constant',
    )

    tokenized_datasets = tokenized_datasets.remove_columns(
        ["article", "title", "year", "publication"]
    )

    trainer = Seq2SeqTrainer(
        model,
        args,
        train_dataset=tokenized_datasets["train"].select(range(50)),
        eval_dataset=tokenized_datasets["test"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=tokenize_compute,
    )

    return trainer
