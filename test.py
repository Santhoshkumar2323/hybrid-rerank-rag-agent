import os
from dotenv import load_dotenv
from groq import Groq

# 1. Load the variables directly from the .env file
load_dotenv()

# 2. Grab your variables from os.environ
model_name = os.environ.get("GROQ_MODEL")
api_key = os.environ.get("GROQ_API_KEY")

print(f"--- Environment Test ---")
print(f"Loaded Model ID from .env: {model_name}")

try:
    # 3. Initialize the Groq client
    client = Groq(api_key=api_key)
    
    # 4. Fire a test chat completion request
    print("\nSending test request to Groq...")
    completion = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "user", "content": "Hello! Reply with 'Success' if you can hear me."}
        ]
    )
    
    print("\n--- API Response Success ---")
    print(completion.choices[0].message.content)

except Exception as e:
    print("\n--- Connection Failed ---")
    print(f"Error details: {e}")
