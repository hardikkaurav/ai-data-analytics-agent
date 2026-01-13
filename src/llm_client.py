# src/llm_client.py

import os
import streamlit as st
from dotenv import load_dotenv
from groq import Groq

# 1. Try loading from .env file (Local)
load_dotenv()

def get_api_key():
    # Try getting from Environment Variables (Local .env)
    api_key = os.getenv("GROQ_API_KEY")
    
    # If not found, try getting from Streamlit Secrets (Cloud)
    if not api_key:
        try:
            api_key = st.secrets["GROQ_API_KEY"]
        except:
            return None
    return api_key

# 2. validation
api_key = get_api_key()

if not api_key:
    # This prevents the app from crashing immediately on import,
    # allows us to show a friendly error in the UI instead if needed.
    # But for now, we raise a clear error to the logs.
    raise ValueError("GROQ_API_KEY not found. Check .env (local) or Streamlit Secrets (Cloud).")

client = Groq(api_key=api_key)

def generate_insights(prompt: str) -> str:
    """
    Generate insights using Groq LLM (LLaMA 3.3)
    """
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
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