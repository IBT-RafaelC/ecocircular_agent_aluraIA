import os
from dotenv import load_dotenv
from google import genai

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

client = genai.Client(api_key=api_key)

print("📋 Modelos disponibles para tu API Key:")
print("-" * 40)

try:
    for model in client.models.list():
        print(f"-> {model.name}")
except Exception as e:
    print(f"Error al consultar modelos: {e}")