import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline

# import gradio as gr
# import random
import time

# store starting time 
begin = time.time() 
 
 
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3.5-mini-instruct",
    torch_dtype=torch.bfloat16,
    # device_map="cpu",
    quantization_config=bnb_config,
    trust_remote_code=True
)
# model.load_adapter('./results/checkpoint-100')

# tokenizer = AutoTokenizer.from_pretrained('./results/checkpoint-100', trust_remote_code=True)
# tokenizer.pad_token = tokenizer.unk_token
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3.5-mini-instruct", trust_remote_code=True)

infstart = time.time()

model_inputs = tokenizer(
    [f"[INST] What is model regularization? [/INST]"], return_tensors="pt", padding=True)
generated_ids = model.generate(**model_inputs, max_new_tokens=60)
result = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

# Run text generation pipeline with our next model
prompt = "What is model regularization?"
pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=200)
result = pipe(f'''[INST] {prompt} [/INST]''')
print(result[0]['generated_text'])

# Run text generation pipeline with our next model
prompt = "What is model regularization?"
pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=200)
result = pipe(f'''{prompt}''')

print(result[0]['generated_text'])

end = time.time()
# total time taken 
print(f"Inference runtime of the program is {end - infstart}")
print(f"Total runtime of the program is {end - begin}")