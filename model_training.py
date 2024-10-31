import torch
from transformers import DistilBertForSequenceClassification, Trainer, TrainingArguments
from utils import ReviewsDataset, tokenize_data
from config import MODEL_PATH, TRAINING_ARGS, DEVICE
from evaluation import compute_metrics
# Load pre-trained model
model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH, num_labels=2).to(DEVICE)

def prepare_datasets(X_train, X_val, y_train, y_val):
    train_encodings = tokenize_data(X_train)
    val_encodings = tokenize_data(X_val)
    train_dataset = ReviewsDataset(train_encodings, y_train)
    val_dataset = ReviewsDataset(val_encodings, y_val)
    return train_dataset, val_dataset

def train_model(train_dataset, val_dataset):
    training_args = TrainingArguments(**TRAINING_ARGS)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics = compute_metrics
    )
    trainer.train()
    return trainer
