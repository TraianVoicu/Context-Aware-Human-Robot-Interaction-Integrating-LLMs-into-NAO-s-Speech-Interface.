import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model

#load model and tokenizatoin
model_name = "facebook/opt-350m"  

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  

# Load model on GPU
device = "cuda" if torch.cuda.is_available() else "cpu"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="cuda",
)

#tokenizare

dataset = load_dataset("json", data_files="naoqi_dataset.jsonl")

def tokenize(batch):
    inputs = [f"Instruction: {x}\nOutput:" for x in batch["instruction"]]
    outputs = [o for o in batch["output"]]
    texts = [i + " " + o for i, o in zip(inputs, outputs)]
    tokenized = tokenizer(texts, truncation=True, padding="max_length", max_length=256)
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

tokenized = dataset.map(tokenize, batched=True)

#apply LORA
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],  # OPT uses same naming
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

training_args = TrainingArguments(
    output_dir="./opt350m_lora_naoqi",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    fp16=True if device=="cuda" else False,
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=20
)

#training
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    tokenizer=tokenizer
)

trainer.train()
trainer.save_model("./opt350m_lora_naoqi")

#inference example
model.eval()
prompt = "Instruction: Turn on the NAOqi robot’s head LEDs.\nOutput:"
inputs = tokenizer(prompt, return_tensors="pt").to(device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        temperature=0.7,
        top_p=0.9,
        do_sample=True
    )

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
