import json
from dotenv import load_dotenv
import os
from google import genai

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

system_prompt = """
You are a helpful assistant who is expert in breaking down complex problems into smaller steps and providing detailed explanations.
For the given user input, analyse the input and break down the problem step by step. Atleast think 5-6 steps on how to solve the problem before solving it down.
Follow steps: analyse → think → output → validate → result

Return STRICT JSON:
{"step":"String","content":"String"}
"""

response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents=[
        {
            "role": "user",
            "parts": [
                {"text": "What is 5 + 5"},
                {"text": "What came first egg or chicken?"}
            ]
        }
    ],
    config={
        "system_instruction": system_prompt,
        "response_mime_type": "application/json"
    }
)

print(response.candidates[0].content.parts[0].text)