from transformers import RobertaConfig, RobertaTokenizer, RobertaForMaskedLM, pipeline
import torch
from transformers import RobertaTokenizer, RobertaConfig, RobertaModel, AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
model = AutoModel.from_pretrained("microsoft/codebert-base")
torch.backends.mps.is_available()