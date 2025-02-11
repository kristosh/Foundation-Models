import pdb

from transformers import AutoModel, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-base")

model = AutoModel.from_pretrained("microsoft/deberta-v3-xsmall")

tokens = tokenizer("Hello world", return_tensors = 'pt')

# process the tokens
pdb.set_trace()
output = model(**tokens)[0]