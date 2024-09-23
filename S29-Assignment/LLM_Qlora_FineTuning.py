import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoTokenizer, pipeline, TrainingArguments
from datasets import load_dataset

# Tried both Phi2 and Phi3.5 models
# model_name = "microsoft/phi-2"
model_name  = "microsoft/Phi-3.5-mini-instruct"

# BitsAndBytes works only on GPU so for CPU you have to use a different quantization library
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)


# Using dataset for openassistant that is converted in chat format
# Original dataset is in normalized format
dataset_name = "timdettmers/openassistant-guanaco"
dataset = load_dataset(dataset_name, split="train")

# Quantization is really important to get a faster output
# Again even for CPU you should figure another quantization like Quanto to help get faster results
# Not much drop in accuracy even with that
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    # device_map="cuda",
    quantization_config=bnb_config,
    trust_remote_code=True
)
model.config.use_cache = False

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
# eos_token is the default but in certain cases unk_token to be used
# tokenizer.pad_token = tokenizer.unk_token

print(model)

from peft import LoraConfig

lora_alpha = 16
lora_dropout = 0.1
lora_r = 64

# Need to ensure the right target modules are selected for proper fine tuning
# PEFT mostly works on the attention weights
peft_config = LoraConfig(
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    r=lora_r,
    bias="none",
    task_type="CAUSAL_LM",
    # target_modules=[
    #     "q_proj",
    #     "k_proj",
    #     "v_proj",
    # ]
    target_modules=[
        "o_proj",
        "qkv_proj",
    ]
)

# Setting up the training arguments
output_dir = "./models"
per_device_train_batch_size = 4
gradient_accumulation_steps = 4
optim = "paged_adamw_32bit"
save_steps = 100
logging_steps = 100
learning_rate = 2e-5
max_grad_norm = 0.3
max_steps = 500
warmup_ratio = 0.03
lr_scheduler_type = "constant"

training_arguments = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    optim=optim,
    save_steps=save_steps,
    logging_steps=logging_steps,
    learning_rate=learning_rate,
    fp16=True,
    max_grad_norm=max_grad_norm,
    max_steps=max_steps,
    warmup_ratio=warmup_ratio,
    group_by_length=True,
    lr_scheduler_type=lr_scheduler_type,
    #gradient_checkpointing=True,
)

from trl import SFTTrainer

max_seq_length = 512

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    args=training_arguments,
)

for name, module in trainer.model.named_modules():
    if "norm" in name:
        module = module.to(torch.float32)

# Training the model
trainer.train()

# Running different tests to check the outputs of the model
# Used pipeline for the same to ensure the other aspects are taken care internally
prompt = "Can you give me an example of python script for Fibonacci series?"
pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=200,device_map="auto")
result = pipe(f"<s>[INST] {prompt} [/INST]")
print(result[0]['generated_text'])

# Run text generation pipeline with our next model
prompt = "How are Sentence Transformers different from Huggingface Transformers?"
pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=200,device_map="auto")
result = pipe(f'''[INST] {prompt} [/INST]''')
print(result[0]['generated_text'])

pipe.model

inputs = tokenizer('''Can you explain what is Contrastive Loss in Deep Learning?''', return_tensors="pt", return_attention_mask=False)
inputs = inputs.to('cuda')

outputs = pipe.model.generate(**inputs, max_length=200)
text = pipe.tokenizer.batch_decode(outputs)[0]
print(text)