import ast
import torch
from transformers import RobertaTokenizer, AutoModelForSequenceClassification

# labels = ["MaxCut", "MIS", "TSP", "Clique", "KColor", "Factor","ADD", "MUL", "SUB", "Unknown"]
# Define problem type tags
PROBLEM_TAGS = {
    "MaxCut": 0,  # Maximum Cut Problem
    "MIS": 1,  # Maximal Independent Set
    "TSP": 2,  # Traveling Salesman Problem
    "Clique": 3,  # Clique Problem
    "KColor": 4,  # K-Coloring
    "Factor": 5,  # Factorization
    "ADD": 6,  # Addition
    "MUL": 7,  # Multiplication
    "SUB": 8,  # Subtraction
    "Unknown": 9
}
GRAPH_TAGS = ["MaxCut", "MIS", "TSP", "Clique", "KColor"]
ARITHMETIC_TAGS = ["ADD", "MUL", "SUB"]

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
        # get a data sets here, then workflow
        # to see if any calls from function calls arguments are suitable
        # for example for {'nx.add_edges_from': [[[1, 2, 3], [4, 5, 6], [7, 8, 9]], 2, 3, 'matrix']}
        # we check [[1, 2, 3], [4, 5, 6], [7, 8, 9]] first, if it's not what we are looking for, then
        # go down to 'matrix' which is a name of a variable, then go to variables sets: var = variables.get(name)
        # to see if var is the correct format, if not, go to variables sets, iterate to see if any vars suit well
        # to check if var is a correct format: for graph problems, an edges like: or distance matrix like [[...]] are fine
        # Return the predicted problem type and None for data (for now)
        # if no variables are suitable to proceed, rase an error.
        :param classical_code: str - The input classical code snippet
        :return: problem_type: str, data: any - Returns the identified problem type and associated data
        """
        # predict labels of problem
        prediction = self._predict_classical_code(classical_code=classical_code)
        problem_class = self._recognize_problem_class(prediction)
        # ast traverse and extract data
        visitor = CodeVisitor()
        tree = ast.parse(classical_code)
        visitor.visit(tree)
        extracted_data = visitor.get_extracted_data()
        # Use extracted data for specific problem types (e.g., graph-related or arithmetic problems)
        if problem_class == "GRAPH":
            data = self._process_graph_data(extracted_data, visitor.function_calls)
        elif problem_class == "ARITHMETIC":
            data = self._process_arithmetic_data(extracted_data, visitor.function_calls)
        else:
            data = None
        return PROBLEM_POOLS[prediction], data


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

    def _recognize_problem_class(self, prediction):
        if PROBLEM_POOLS[prediction] in GRAPH_TAGS:
            return "GRAPH"
        elif PROBLEM_POOLS[prediction] in ARITHMETIC_TAGS:
            return "ARITHMETIC"
        return "UNKNOWN"

    def _process_graph_data(self, variables, function_calls):
        """
        Process graph-related data, such as extracting edges from function calls like nx.add_edges_from.
        """
        # Check for function calls like nx.add_edges_from
        for func_name, args in function_calls.items():
            if "add_edges_from" in func_name:
                return args[0]  # Assume the first argument is the edge list
        # Return extracted variables, if no suitable function call found
        return variables

    def _process_arithmetic_data(self, variables, function_calls):
        """
        Process arithmetic-related data, such as extracting operands from addition, multiplication, etc.
        """
        return variables

class CodeVisitor(ast.NodeVisitor):
    def __init__(self):
        self.variables = {}
        self.function_calls = {}

    def visit_Assign(self, node):
        """
        Capture assignments in the code, e.g., variables or data structures.
        """
        print(ast.dump(node))
        for target in node.targets:
            if isinstance(target, ast.Tuple):
                self._process_tuple(target, node.value)
            if isinstance(target, ast.Name):
                value_repr = self._process_value(node.value)
                self.variables[target.id] = value_repr
        self.generic_visit(node)  # Continue traversing

    def visit_Call(self, node):
        """
        Capture function calls, including object method calls like nx.add_edges_from.
        """
        func_name = self._get_function_name(node.func)
        # Process arguments of the function call
        args = [self._process_value(arg) for arg in node.args]
        # Store the function call details
        self.function_calls[func_name] = args

        self.generic_visit(node)  # Continue traversing other nodes

    def _get_function_name(self, func):
        """
        Get the function name, whether it's a direct function call or an attribute-based method call (e.g., nx.add_edges_from).
        :param func: The AST node representing the function.
        :return: The full function name as a string.
        """
        if isinstance(func, ast.Attribute):  # Handle method calls like nx.add_edges_from
            # Get the object and method name
            obj_name = func.value.id if isinstance(func.value, ast.Name) else "unknown_obj"
            return f"{obj_name}.{func.attr}"
        elif isinstance(func, ast.Name):  # Handle simple function calls like add(a, b)
            return func.id
        return "unknown_func"

    def _process_tuple(self, target, value):
        """
        Process tuple assignments like (p, q) = (8, 8).
        :param target: The tuple on the left-hand side.
        :param value: The value being assigned (could also be a tuple).
        """
        if isinstance(value, ast.Tuple):
            for i, elt in enumerate(target.elts):
                if isinstance(elt, ast.Name) and i < len(value.elts):
                    self.variables[elt.id] = self._process_value(value.elts[i])

    def _process_value(self, value):
        """
        Process the value being assigned to variables. This could be a constant, variable, or function call.
        :param value: The value being assigned (right-hand side of an assignment).
        :return: The processed value representation.
        """
        # If the value is a constant, return the constant's value
        if isinstance(value, ast.Constant):
            return value.value

        # If the value is a variable, return the variable's name
        elif isinstance(value, ast.Name):
            return value.id

        # If the value is a function call, return the function name and arguments
        elif isinstance(value, ast.Call):
            func_name = value.func.id if isinstance(value.func, ast.Name) else "unknown_func"
            args = [self._process_value(arg) for arg in value.args]
            return f"Function Call: {func_name}({', '.join(map(str, args))})"

        # If the value is a list, process each element
        elif isinstance(value, ast.List):
            return [self._process_value(elt) for elt in value.elts]

        # If the value is a tuple, process each element
        elif isinstance(value, ast.Tuple):
            return tuple(self._process_value(elt) for elt in value.elts)
        return "Unknown Value"

    def get_extracted_data(self):
        """
        Return the extracted data from the AST traversal.
        :return: dict of extracted variables.
        """
        return self.variables
