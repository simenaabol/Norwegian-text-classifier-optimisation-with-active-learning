from scipy.stats import entropy
from modAL.utils.selection import shuffled_argmax
from transformers import Trainer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score)
from datasets import load_dataset, concatenate_datasets, Dataset
from transformers import (DataCollatorWithPadding, AutoModelForSequenceClassification,
                          logging, TrainingArguments, AutoTokenizer)
import tensorflow as tf
import pandas as pd
import numpy as np
import csv
import torch
import os
from config import huggingface
# from utils import print_gpu_utilization
import re
from decimal import Decimal


print("-------------- Kan vi bruke CUDA? ---------------")
print(torch.cuda.is_available())

logging.set_verbosity_error()

#  Burde mulig aktiveres senere for å forbedre kjøretid?
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# print_gpu_utilization()

torch.cuda.synchronize()


def tokenize_dataset(dataset: Dataset, model_checkpoint: str) -> Dataset:
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    train_data = dataset["train"]
    val_data = dataset["validation"]
    test_data = dataset["test"]

    train_data = train_data.map(tokenize_function, batched=True)
    # train_data = train_data.with_format("torch")

    val_data = val_data.map(tokenize_function, batched=True)
    # val_data = val_data.with_format("torch")

    test_data = test_data.map(tokenize_function, batched=True)
    # test_data = val_data.with_format("torch")

    return train_data, val_data, test_data


dataset = load_dataset("NTCAL/reviews_binary_not4_concat")
dataset = dataset.remove_columns(
    ["split", "review_id", "year", "category", "language", "title"])
dataset = dataset.rename_column("excerpt", "text")
dataset = dataset.rename_column("rating", "label")

model_checkpoint = ("ltgoslo/norbert2")
train_data, val_data, test_data = tokenize_dataset(
    dataset, model_checkpoint)

# test_concat = concatenate_datasets([val_data, test_data])
# print(len(test_concat))

default_args = {
    "output_dir": "./results",
    "overwrite_output_dir": True,
    "log_level": "error",
    "report_to": "none",
}

training_args = TrainingArguments(
    num_train_epochs=5,              # total number of training epochs
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    gradient_accumulation_steps=8,
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=10,
    evaluation_strategy="epoch",
    # evaluation_strategy="steps",
    # eval_steps=10,
    # warmup_steps=1,                # number of warmup steps for learning rate scheduler
    gradient_checkpointing=True,
    optim="adafactor",
    save_strategy="epoch",
    learning_rate=5e-5,

    # load_best_model_at_end=True

    # hub_token=huggingface['hub_token'],
    # push_to_hub=True,
    # hub_model_id=huggingface['repo'],

    # fp8=True,
    # fp16=True,
    # bf16=True,
    tf32=True,

    **default_args,
)


def pick_n_random_reviews(train_data: Dataset, n: int):
    random_indices = np.random.choice(
        list(range(0, len(train_data))), n, replace=False)
    random_reviews = train_data[random_indices]

    train_data = train_data.select(
        (
            i for i in range(len(train_data))
            if i not in set(random_indices)
        )
    )

    return train_data, random_reviews


def create_dataset_from_empty_df() -> Dataset:
    empty_df = pd.DataFrame(
        columns=['text', 'label', 'input_ids', 'token_type_ids', 'attention_mask'])
    empty_df.reset_index(drop=True, inplace=True)
    empty_ds = Dataset.from_pandas(empty_df)

    return empty_ds


def add_samples_to_pool(train_pool: Dataset, new_reviews: dict) -> Dataset:
    list_of_reviews_dicts = [{'text': text, 'label': label, 'input_ids': input_ids, 'token_type_ids': token_type_ids,
                              'attention_mask': attention_mask} for text, label, input_ids, token_type_ids, attention_mask in zip(*new_reviews.values())]

    for review in list_of_reviews_dicts:
        train_pool = train_pool.add_item(review)

    return train_pool


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    # Calculate accuracy using sklearn's function
    acc = accuracy_score(labels, preds)
    balanced_accuracy = balanced_accuracy_score(labels, preds)
    f1 = f1_score(labels, preds)
    recall = recall_score(labels, preds)
    precision = precision_score(labels, preds)

    accuracy_dict = {
        'accuracy': acc,
        'balanced_accuracy': balanced_accuracy,
        'f1_score': f1,
        'recall': recall,
        'precision': precision,
    }

    # Write accuracy metrics to csv file
    # with open(CSV_FILE_NAME, 'a') as f:
    #     w = csv.DictWriter(f, fieldnames=[
    #                        "accuracy", "balanced_accuracy", "f1_score", "recall", "precision"])
    #     w.writerow(accuracy_dict)

    return accuracy_dict


def get_model_util(model_checkpoint: str):
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_checkpoint).to("cuda")

    return tokenizer, data_collator, model


def least_confident_sampling(train_data: Dataset, n: int, model: Trainer):
    # From https://github.com/Kantega-AI-team/advanced-ml-concepts/blob/master/Active%20Learning.ipynb

    predictions = model.predict(train_data)
    preds = tf.nn.softmax(predictions.predictions)
    preds = preds.numpy()

    df = pd.DataFrame(preds, columns=[0, 1])
    df['max'] = df.max(axis=1)

    least_confident_predictions = df.nsmallest(n, columns=['max'])
    index_of_least_confident = least_confident_predictions.index.values

    new_reviews = train_data[index_of_least_confident]

    train_data = train_data.select(
        (
            i for i in range(len(train_data))
            if i not in set(index_of_least_confident)
        )
    )

    return train_data, new_reviews


def margin_sampling(train_data: Dataset, n: int, model: Trainer):
    # From https://github.com/modAL-python/modAL/blob/master/modAL/uncertainty.py

    predictions = model.predict(train_data)
    preds = tf.nn.softmax(predictions.predictions)
    preds = preds.numpy()

    part = np.partition(-preds, 1, axis=1)
    margin = - part[:, 0] + part[:, 1]
    margin_indices = shuffled_argmax(-margin, n_instances=n)

    new_reviews = train_data[margin_indices]

    train_data = train_data.select(
        (
            i for i in range(len(train_data))
            if i not in set(margin_indices)
        )
    )

    return train_data, new_reviews


def entropy_sampling(train_data: Dataset, n: int, model: Trainer):
    # From https://github.com/modAL-python/modAL/blob/master/modAL/uncertainty.py

    predictions = model.predict(train_data)
    preds = tf.nn.softmax(predictions.predictions)
    preds = preds.numpy()

    entropy_out = np.transpose(entropy(np.transpose(preds), base=2))
    entropy_indices = shuffled_argmax(entropy_out, n_instances=n)

    new_reviews = train_data[entropy_indices]

    train_data = train_data.select(
        (
            i for i in range(len(train_data))
            if i not in set(entropy_indices)
        )
    )

    return train_data, new_reviews


def train_AL(n: int, model_checkpoint: str, active_learning: bool, type_sampling: str, train_new_models: bool):

    # Create csv file to hold accuracy metrics
    with open(CSV_FILE_NAME, 'w') as f:
        w = csv.DictWriter(f, fieldnames=[
            'eval_runtime', 'eval_balanced_accuracy', 'epoch', 'eval_f1_score', 'eval_loss', 'eval_recall', 'eval_precision', 'eval_samples_per_second', 'eval_accuracy', 'eval_steps_per_second', 'train_runtime', 'train_samples_per_second', 'train_steps_per_second', 'train_loss'])
        w.writeheader()

    tokenizer, data_collator, model = get_model_util(model_checkpoint)

    train_set = train_data
    val_set = val_data
    test_set = test_data

    # Pick n amount of reviews to start training on, and remove them from the training data
    train_set, new_reviews = pick_n_random_reviews(train_set, n)

    # Initialize empty training pool, and add the found reviews to the training pool
    # TODO this should "represent" the whole dataset. Maybe set it somewhat manually
    train_pool = create_dataset_from_empty_df()
    train_pool = add_samples_to_pool(train_pool, new_reviews)
    print("pool", train_pool)

    while len(train_set) > n:

        print("START OF LOOP. NR OF ROWS IN TRAIN SET:", len(train_set))

        model.resize_token_embeddings(len(tokenizer))
        trainer = Trainer(model=model,
                          args=training_args,
                          train_dataset=train_pool,
                          data_collator=data_collator,
                          tokenizer=tokenizer,
                          eval_dataset=val_set,
                          compute_metrics=compute_metrics
                          )

        # Train model on chosen reviews
        trainer_results = trainer.train()

        # The evaluation metrics from on test set
        score = trainer.evaluate(test_set)
        score.update(trainer_results.metrics)

        with open(CSV_FILE_NAME, 'a') as f:
            w = csv.DictWriter(f, fieldnames=[
                'eval_runtime', 'eval_balanced_accuracy', 'epoch', 'eval_f1_score', 'eval_loss', 'eval_recall', 'eval_precision', 'eval_samples_per_second', 'eval_accuracy', 'eval_steps_per_second', 'train_runtime', 'train_samples_per_second', 'train_steps_per_second', 'train_loss'])
            w.writerow(score)

        if active_learning:
            # Get the n indices depending on the active learning sampling method
            if type_sampling == "least_certain":
                print(f"{type_sampling} sampling started")
                train_set, new_reviews = least_confident_sampling(
                    train_set, n, trainer)

            elif type_sampling == "margin":
                print(f"{type_sampling} sampling started")
                train_set, new_reviews = margin_sampling(
                    train_set, n, trainer)

            elif type_sampling == "entropy":
                print(f"{type_sampling} sampling started")
                train_set, new_reviews = entropy_sampling(
                    train_set, n, trainer)

            else:
                ValueError("Choose a valid uncertainty sampling method")

        else:
            # Get the n indices randomly
            train_set, new_reviews = pick_n_random_reviews(train_set, n)

        if train_new_models:
            # Get a new non-fine-tuned model and expand the training pool
            del trainer
            tokenizer, data_collator, model = get_model_util(
                model_checkpoint)
            train_pool = add_samples_to_pool(train_pool, new_reviews)
        else:
            # Keep the same model, and only keep new rows for training
            train_pool = create_dataset_from_empty_df()
            train_pool = add_samples_to_pool(train_pool, new_reviews)


def train_model(model_checkpoint: str):

    # Create csv file to hold accuracy metrics
    with open(CSV_FILE_NAME, 'w') as f:
        w = csv.DictWriter(f, fieldnames=[
            "accuracy", "balanced_accuracy", "f1_score", "recall", "precision"])
        w.writeheader()

    tokenizer, data_collator, model = get_model_util(model_checkpoint)

    model.resize_token_embeddings(len(tokenizer))
    trainer = Trainer(model=model,
                      args=training_args,
                      train_dataset=train_data,
                      data_collator=data_collator,
                      tokenizer=tokenizer,
                      eval_dataset=val_data,
                      compute_metrics=compute_metrics
                      )

    # Train model
    trainer.train()


CSV_FILE_NAME = "entropyshannon_200p32b5epo_iter.csv"

train_AL(n=200, model_checkpoint="ltg/norbert2", active_learning=False,
         type_sampling="entropy", train_new_models=False)
# train_model("ltg/norbert2")

# TODO for hvert run:
# 1. Velg riktig CSV_FILE_NAME
# 2. Velg riktig metode train_AL eller train_model, med riktig parametre
# 3. Dobbeltsjekk training_args
# 4. Skriv samme navn som CSV_FILE_NAME i output file i IDUN
