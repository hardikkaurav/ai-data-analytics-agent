# src/llm_client.py

import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

# Ensure the API key exists to avoid crashing
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY not found. Please check your .env file.")

client = Groq(api_key=api_key)

def generate_insights(prompt: str) -> str:
    """
    Generate insights using Groq LLM (LLaMA 3.3)
    """
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",   # UPDATED: The new supported model
            messages=[
                {"role": "system", "content": "You are a professional data analyst."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=512
        )
        return response.choices[0].message.content
        
    except Exception as e:
        return f"Error generating insights: {e}"