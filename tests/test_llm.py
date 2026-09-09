import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"


print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype="auto",
    device_map="auto"
)

print("Model loaded successfully.")

messages = [
    {
        "role": "user",
        "content": "Jelaskan secara singkat apa itu Vision-Language Model."
    }
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

model_inputs = tokenizer(
    [text],
    return_tensors="pt"
).to(model.device)

print("Generating response...")

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=200
)

generated_ids = [
    output_ids[len(input_ids):]
    for input_ids, output_ids in zip(
        model_inputs.input_ids,
        generated_ids
    )
]

response = tokenizer.batch_decode(
    generated_ids,
    skip_special_tokens=True
)[0]

print("\n===== MODEL RESPONSE =====")
print(response)
print("==========================")