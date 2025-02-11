import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

import pdb

tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")

model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3-mini-4k-instruct",
    device_map = "cpu",
    torch_dtype = "auto",
    trust_remote_code = True
)

# Create a pipeline
generator = pipeline(
    "text-generation",
    model = model,
    tokenizer = tokenizer,
    return_full_text= False,
    max_new_tokens = 50,
    do_sample = False
)

prompt = "the capital of France is"

input_ids = tokenizer(prompt, return_tensors = "pt").input_ids

# tokenize the input prompt
input_ids = input_ids.to("cpu")

# get the output of the model
model_output = model.model(input_ids)

lm_head_output = model.lm_head(model_output[0])

token_id = lm_head_output[0,-1].argmax(-1)

pdb.set_trace()
print(tokenizer.decode(token_id))

logits = lm_head_output[0, -1]

# Get the top 4 token indices and their corresponding probabilities
top_k = 40
top_probs, top_indices = torch.topk(logits, k=top_k, dim=-1)


print(tokenizer.decode(top_indices))

