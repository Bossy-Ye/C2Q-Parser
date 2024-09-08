from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch

# Load the tokenizer and model (CodeBERT is based on RoBERTa)
tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
model = RobertaForSequenceClassification.from_pretrained("microsoft/codebert-base", num_labels=3)  # 3 labels for problem types

# Example code snippet
code = """
def knapsack(weights, values, capacity):
    dp = [[0] * (capacity + 1) for _ in range(len(weights) + 1)]
    for i in range(1, len(weights) + 1):
        for w in range(1, capacity + 1):
            if weights[i - 1] <= w:
                dp[i][w] = max(dp[i - 1][w], dp[i - 1][w - weights[i - 1]] + values[i - 1])
            else:
                dp[i][w] = dp[i - 1][w]
    return dp[-1][-1]
"""

# Tokenize the input code
inputs = tokenizer(code, return_tensors="pt")

# Predict the problem type
outputs = model(**inputs)
predicted_class = torch.argmax(outputs.logits, dim=-1)
print(f"Predicted problem type: {predicted_class.item()}")