import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    print("No API Key found in .env")
else:
    try:
        genai.configure(api_key=api_key)
        print("Listing models...")
        found = False
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                print(f"- {m.name}")
                found = True
        if not found:
            print("No generateContent models found.")
    except Exception as e:
        print(f"Error: {e}")
