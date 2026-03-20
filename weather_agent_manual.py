import json
from dotenv import load_dotenv
import os
from google import genai
from google.genai import types

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


def get_weather(city: str):
    return "31 degree celsius"


system_prompt = """
You are an helpful AI assistant who is specialised in solving user query 
You work on start, plan, action, observe mode.
For the given user query and available tools, plan the step by step execution.
Based on the planning, select the relevant tool from the available tools.
Based on the tool selection, perform an action to call the tool.
Wait for the observation and based on the observation from the tool call, solve the user query.

Rules:
- Follow the output JSON format strictly
- Always perform one step at a time and wait for the next input
- Carefully analyse the user query

Output format:
{
  "step": "String",
  "content": "String",
  "function": "The name of function if the step is action",
  "input": "The input parameter for the function"
}

Available Tools:
- get_weather: Takes a city name as string input and returns the current weather of that city

Example:
User Query: What is the weather of New York?
Output: {"step": "plan", "content": "The user is interested in weather data of New York"}
Output: {"step": "plan", "content": "From the available tools I should call get_weather"}
Output: {"step": "action", "function": "get_weather", "input": "New York"}
Output: {"step": "observation", "output": "12 degree celsius"}
Output: {"step": "output", "content": "The weather of New York seems to be 12 degree celsius"}
"""

contents = [
    {
        "role": "user",
        "parts": [{"text": "what is the current weather in patiala?"}]
    },
    {
        "role": "model",
        "parts": [{"text": json.dumps({"step": "plan", "content": "The user is interested in weather data of Patiala"})}]
    },
    {
        "role": "model",
        "parts": [{"text": json.dumps({"step": "plan", "content": "From the available tools I should call get_weather"})}]
    },
    {
        "role": "model",
        "parts": [{"text": json.dumps({"step": "action", "function": "get_weather", "input": "Patiala"})}]
    },
    {
        "role": "model",
        "parts": [{"text": json.dumps({"step": "observation", "output": "31 degree celsius"})}]
    },
]

response = client.models.generate_content(
    model="gemini-2.0-flash",
    config=types.GenerateContentConfig(
        system_instruction=system_prompt,
        response_mime_type="application/json",  # replaces response_format
    ),
    contents=contents,
)

print(response.candidates[0].content.parts[0].text)