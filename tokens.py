from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3-mini-4k-instruct",
    device_map = "cuda",
    torch_dtype = "auto",
    trust_remote_code = True
)

tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")

prompt = "Writing an email apologizing to Chris for the dealy in submitting the thesis on time. Explain why it happened.<|assistant|>"

#tokenize the input prompt 
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cpu")

#generating the text
generation_output = model.generate(
    input_ids = input_ids,
    max_new_tokens = 158
)

print(tokenizer.decode(generation_output[0]))