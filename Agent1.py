import json
from dotenv import load_dotenv
import requests
import os
from google import genai
from google.genai import types
import subprocess
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


def get_weather(city: str):
    print(f"🔨 Tool called: get_weather({city})")
    url = f"https://wttr.in/{city}?format=%C+%t"
    response = requests.get(url)
    if response.status_code == 200:
        return f"the current weather in {city} is {response.text}"
    return "Something went wrong"

def run_command(command: str):
    print(f"🔨 Tool called: run_command({command})")
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    return result.stdout.strip() or result.stderr.strip() or "No output"

available_tools = {
    "get_weather": {
        "function": get_weather,
        "description": "Takes a city name as string input and returns the current weather of that city"
    },

    "run_command":{
        "function": run_command,
        "description": "Takes a command as string input and runs it in the terminal, returns the result"
    }
}

system_prompt = """
You are a helpful AI assistant who specialises in solving user queries.
You work in start, plan, action, observe mode.
For the given user query and available tools, plan the step by step execution.
Based on the planning, select the relevant tool from the available tools.
Based on the tool selection, perform an action to call the tool.
Wait for the observation and based on the observation from the tool call, solve the user query.

Rules:
- Follow the output JSON format strictly
- Always perform one step at a time and wait for the next input
- Carefully analyse the user query
- Always use lowercase for the "step" field value

Output format:
{
  "step": "string",
  "content": "string",
  "function": "the name of function if the step is action",
  "input": "the input parameter for the function"
}

Available Tools:
- get_weather: Takes a city name as string input and returns the current weather of that city
- run_command: Takes a command as string input and runs it in the terminal, returns the result

Example:
User Query: What is the weather of New York?
Output: {"step": "plan", "content": "The user is interested in weather data of New York"}
Output: {"step": "plan", "content": "From the available tools I should call get_weather"}
Output: {"step": "action", "function": "get_weather", "input": "New York"}
Output: {"step": "observation", "output": "12 degree celsius"}
Output: {"step": "output", "content": "The weather of New York seems to be 12 degree celsius"}
"""

contents = []

while True:
    user_input = input("Enter your query: ")
    contents.append({
        "role": "user",
        "parts": [{"text": user_input}]
    })

    while True:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                response_mime_type="application/json",
            ),
            contents=contents, #passes the history of conversation to the model for better context
        )

        parsed_output = json.loads(response.candidates[0].content.parts[0].text)

        # Append model's response in Gemini format
        contents.append({
            "role": "model",
            "parts": [{"text": json.dumps(parsed_output)}]
        })

        if parsed_output.get("step") == "plan":
            print(f"🧠 {parsed_output.get('content')}")
            continue

        if parsed_output.get("step") == "action":
            tool_name = parsed_output.get("function")
            tool_input = parsed_output.get("input")
            print(f"🔧 Calling: {tool_name}({tool_input})")

            if tool_name in available_tools:
                tool_output = available_tools[tool_name]["function"](tool_input)
            else:
                tool_output = f"Error: tool '{tool_name}' not found"

            # ✅ Correct Gemini format — role must be "user", not "assistant"
            observation = json.dumps({"step": "observation", "output": tool_output})
            contents.append({
                "role": "user",
                "parts": [{"text": observation}]
            })
            continue

        if parsed_output.get("step") == "output":
            print(f"🤖 {parsed_output.get('content')}")
            break