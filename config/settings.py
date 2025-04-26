import os
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

#GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
#print(GOOGLE_API_KEY)
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
if not GOOGLE_API_KEY:
    print("Warning: GOOGLE_API_KEY not found in environment variables. Please ensure it's set, e.g., in a .env file.")

# Get the Gemini Model Name from environment variables, with a default
#GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-2.0-flash")
GEMINI_MODEL_NAME = st.secrets["GEMINI_MODEL_NAME"]
