# chatbot.py

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Model name - Use the Instruct-tuned version
model_name = "Qwen/Qwen2-0.5B-Instruct"

# Load tokenizer and model once at startup
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)

# Set pad_token_id to eos_token_id for open-ended generation
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = model.config.eos_token_id


# Function to generate chatbot response (Final Corrected Version)
def chatbot(prompt: str) -> str:
    # Use a system prompt for better, more consistent behavior
    messages = [
        {"role": "system", "content": "You are a helpful and friendly chatbot."},
        {"role": "user", "content": prompt}
    ]
    
    # Step 1: Format the chat messages into a single string using the template.
    # The `add_generation_prompt=True` is crucial for instruct models.
    prompt_string = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )

    # Step 2: Tokenize the formatted string. This reliably returns a dictionary
    # containing 'input_ids' and 'attention_mask'.
    model_inputs = tokenizer(prompt_string, return_tensors="pt").to(model.device)

    # Generate the response
    with torch.no_grad():
        # The **model_inputs call now works perfectly because model_inputs is a dictionary.
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )

    # The generated IDs contain both the input prompt and the response.
    # We need to decode only the newly generated tokens.
    response_ids = generated_ids[0][model_inputs.input_ids.shape[1]:]
    response = tokenizer.decode(response_ids, skip_special_tokens=True)
    
    return response

# FastAPI app
app = FastAPI()

# Enable CORS (allow frontend to call backend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    user_input = data.get("message", "")
    if not user_input:
        return {"response": "Please provide a message."}
    
    response = chatbot(user_input)
    return {"response": response}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)