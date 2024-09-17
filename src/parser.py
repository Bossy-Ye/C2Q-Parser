import ast
import torch
from transformers import RobertaTokenizer, AutoModelForSequenceClassification
# labels = ["MaxCut", "MIS", "TSP", "Clique", "KColor", "Factor","ADD", "MUL", "SUB", "Unknown"]
# Define problem type tags
PROBLEM_TAGS = {
    "MaxCut": 0, # Maximum Cut Problem
    "MIS": 1,  # Maximal Independent Set
    "TSP": 2,  # Traveling Salesman Problem
    "Clique": 3,    # Clique Problem
    "KColor": 4,  # K-Coloring
    "Factor": 5,  # Factorization
    "ADD": 6,   # Addition
    "MUL": 7,   # Multiplication
    "SUB": 8,   # Subtraction
    "Unknown": 9
}

# Reverse mapping, e.g., PROBLEM_POOLS[2] = "TSP"
PROBLEM_POOLS = [k for k, v in PROBLEM_TAGS.items()]


class Parser:
    def __init__(self, model_path: str):
        """
        Initialize the parser with the tokenizer, model, and device.

        :param model_path: str - Path to the saved model directory
        """
        self.device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        self.tokenizer = RobertaTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)

    def parse(self, classical_code: str):
        """
        Parse the classical code, determine the problem type and extract relevant data.

        :param classical_code: str - The input classical code snippet
        :return: problem_type: str, data: any - Returns the identified problem type and associated data
        """
        # predict labels of problem
        prediction = self._predict_classical_code(classical_code=classical_code)
        print(prediction)
        # ast traverse
        data = self._extract_data(classical_code=classical_code,
                                  prediction=prediction)
        # Return the predicted problem type and None for data (for now)
        return PROBLEM_POOLS[prediction], data

    def _extract_data(self, classical_code: str, prediction):
        visitor = CodeVisitor()
        return 1

    def _predict_classical_code(self, classical_code: str):
        """
        Tokenize the input code, pass it through the model, and return the predicted problem type index.

        :param classical_code: str - The input classical code snippet
        :return: int - The predicted problem type index
        """
        inputs = self.tokenizer(classical_code, return_tensors="pt", padding="max_length", truncation=True).to(
            self.device)
        outputs = self.model(**inputs)
        probabilities = torch.softmax(outputs.logits, dim=-1)  # Get probabilities
        max_prob, prediction = torch.max(probabilities, dim=-1)
        return prediction.item()


class CodeVisitor(ast.NodeVisitor):
    def __init__(self):
        self.variables = {}

    def visit_Assign(self, node):
        """
        Capture assignments in the code, e.g., variables or data structures.
        """
        for target in node.targets:
            if isinstance(target, ast.Name):
                self.variables[target.id] = ast.dump(node.value)
        self.generic_visit(node)  # Continue traversing

    def get_extracted_data(self):
        """
        Return the extracted data from the AST traversal.
        :return: dict of extracted variables.
        """
        return self.variables


# Example usage:
if __name__ == "__main__":
    # Initialize the parser with the path to the saved model
    model_path = "saved_models"
    parser = Parser(model_path)

    # Sample code snippet
    code_snippet = "def a(p, q):\n    return p * q\n\n# Input data\np, q = 8, 8\nresult = a(p, q)\nprint(result)"
    # Parse the code snippet
    problem_type, data = parser.parse(code_snippet)
    tree = ast.parse(code_snippet)
    visitor = CodeVisitor()
    visitor.visit(tree)
    print(f"Problem Type: {problem_type}, Data: {data}")
