import numpy as np
from datasets import load_dataset, load_metric
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments, AutoTokenizer, \
    AutoModelForSequenceClassification
import ast

# Load my dataset
dataset = load_dataset('csv', data_files={'train': 'train.csv', 'test': 'test.csv'})
print(dataset['train'][12])
module = ast.parse(dataset['train'][1].get('code_snippet'))
module = ast.parse(dataset['train'][35].get('code_snippet'))
# CodeBERT tokenizer
tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")


def tokenize_function(examples):
    code_snippet = examples['code_snippet']
    return tokenizer(code_snippet, padding="max_length", truncation=True)


tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Access the tokenized input (input_ids) for a specific sample
token_ids = tokenized_datasets['train'][13]["input_ids"]

# Convert token IDs back to tokens
tokens = tokenizer.convert_ids_to_tokens(token_ids)

# Print the tokens
print(tokens)

model = AutoModelForSequenceClassification.from_pretrained("microsoft/codebert-base", num_labels=5)
training_args = TrainingArguments(output_dir="test_trainer",
                                  eval_strategy="epoch",
                                  num_train_epochs=10)

train_dataset = tokenized_datasets['train']
eval_dataset = tokenized_datasets['test']

# # Load the metric you want to compute
# metric = load_metric("accuracy")  # You can also use "f1", "precision", etc.
#
#
# # Define a compute_metrics function for accuracy
# def compute_metrics(eval_pred):
#     logits, labels = eval_pred
#     predictions = np.argmax(logits, axis=-1)
#     return metric.compute(predictions=predictions, references=labels)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

trainer.train()
results = trainer.evaluate()
print(results)

import torch

print(torch.backends.mps.is_available())  # Should return True if MPS is supported.
print(torch.backends.mps.is_built())  # Should return True if PyTorch is built with MPS support.
device = torch.device("cpu")
model.to(device)
code = """
import networkx as nx
import random

def maximal_independent_set(G):
    # Get all the nodes of the graph
    nodes = set(G.nodes())
    
    # Initialize an empty set for the independent set
    independent_set = set()
    
    # While there are still nodes left to process
    while nodes:
        # Randomly select a node
        v = random.choice(list(nodes))
        
        # Add the node to the independent set
        independent_set.add(v)
        
        # Remove the node and its neighbors from the set of available nodes
        neighbors = set(G.neighbors(v))
        nodes = nodes - neighbors - {v}
    
    return independent_set


"""

labels = ["MaxCut", "MIS", "TSP", "Clique", "KColor", "Unknown"]


def predict(code: str):
    inputs = tokenizer(code, return_tensors="pt").to(device)
    outputs = model(**inputs)
    probabilities = torch.softmax(outputs.logits, dim=-1)  # Get probabilities
    max_prob, prediction = torch.max(probabilities, dim=-1)

    return labels[prediction.item()]


print(len(eval_dataset))
for i in range(len(eval_dataset)):
    code = eval_dataset[i].get('code_snippet')
    print(predict(code))
