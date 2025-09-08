import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

#load base model and LORA

base_model_name = "facebook/opt-350m"
lora_model_path = "./opt350m_lora_naoqi"

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(base_model_name)
tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    device_map="auto" if device=="cuda" else None
)

model = PeftModel.from_pretrained(base_model, lora_model_path)
model.eval()

#loop for chatting
print("Type 'exit' to quit.\n")

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break

    prompt = f"Instruction: {user_input}\nOutput:"

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Remove the instruction part from the output
    
    response = response.split("Output:")[-1].strip()
    print("Model:", response)
