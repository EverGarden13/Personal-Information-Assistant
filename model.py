from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json

# Set up device for GPU usage
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the tokenizer and model
model_name = 'meta-llama/Llama-3.2-3B'  # Model version
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Add a pad token if it doesn't exist
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

# Load personal information from JSON
with open('personal_info.json', 'r') as f:
    data = json.load(f)

personal_information = data['personal_info'][0]  # Assuming you have one main profile
system_prompt = personal_information['system_prompt']  # Load the system prompt from JSON

def generate_reply(prompt):
    # Prepare additional information to include in the response
    skills_info = f"My skills include: {', '.join(personal_information['skills'])}."
    interests_info = f"My interests are: {', '.join(personal_information['interests'])}."
    
    # Combine the system prompt, user input, and additional information
    full_prompt = f"System: {system_prompt}\n"
    full_prompt += f"{skills_info} {interests_info}\n"
    full_prompt += f"User: {prompt}\nAssistant:"

    # Encode the full prompt
    inputs = tokenizer(full_prompt, return_tensors='pt', padding=True).to(device)
    
    # Generate a response from the model
    with torch.no_grad():
        outputs = model.generate(
            inputs['input_ids'], 
            attention_mask=inputs['attention_mask'],  # Provide attention mask
            max_length=200, 
            temperature=0.7, 
            top_p=0.9, 
            do_sample=True,  # Set do_sample to True
            pad_token_id=tokenizer.eos_token_id
        )

    # Decode the generated response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract the assistant's reply (ignore the system prompt)
    assistant_reply = response.split("Assistant:")[-1].strip()
    
    return assistant_reply