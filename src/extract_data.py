from transformers import BertTokenizer, BertForQuestionAnswering
import torch

# Initialize the tokenizer and model (you can also use CodeBERT)
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForQuestionAnswering.from_pretrained("bert-base-uncased")

# Example code snippet
code_snippet = """
weights = [10, 20, 30]
values = [60, 100, 120]
capacity = 50
"""

# Tokenize the code snippet
inputs = tokenizer(code_snippet, return_tensors="pt")

# Sample span positions for training (start and end indices of the relevant data)
start_positions = torch.tensor([5])  # Assume the start of 'weights'
end_positions = torch.tensor([11])   # Assume the end of 'weights'

# Forward pass (for training)
outputs = model(**inputs, start_positions=start_positions, end_positions=end_positions)

# Loss calculation (for training)
loss = outputs.loss
print(f"Training loss: {loss.item()}")



# Tokenize new code snippet
inputs = tokenizer(code_snippet, return_tensors="pt")

# Forward pass (no labels during inference)
outputs = model(**inputs)

# Predicted start and end positions
start_position = torch.argmax(outputs.start_logits)
end_position = torch.argmax(outputs.end_logits)

# Extract the predicted span (relevant data)
tokens = inputs['input_ids'][0][start_position:end_position+1]
extracted_data = tokenizer.decode(tokens)
print(f"Extracted data: {extracted_data}")