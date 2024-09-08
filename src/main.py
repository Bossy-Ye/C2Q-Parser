from datasets import load_dataset
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
import ast

# Load my dataset
dataset = load_dataset('csv', data_files={'train': 'train.csv', 'test': 'test.csv'})
# CodeBERT tokenizer
tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")

# # Tokenize the dataset
# def preprocess_function(examples):
#     return tokenizer(examples['code_snippet'], truncation=True, padding=True)
#
#
# tokenized_datasets = dataset.map(preprocess_function, batched=True)
# # Define the model (ensure that num_labels matches your number of classes)
# model = RobertaForSequenceClassification.from_pretrained("microsoft/codebert-base", num_labels=2)
#
# # Define training arguments
# training_args = TrainingArguments(
#     output_dir='./results',
#     evaluation_strategy="epoch",
#     learning_rate=2e-5,
#     per_device_train_batch_size=8,
#     per_device_eval_batch_size=16,
#     num_train_epochs=3,
#     weight_decay=0.01,
# )
#
# # Initialize Trainer
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=tokenized_datasets['train'],
#     eval_dataset=tokenized_datasets['test']
# )
#
# # Train the model
# trainer.train()