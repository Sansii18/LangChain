from dotenv import load_dotenv
import json
import os
import pathlib
from google import genai
from google.genai import types
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# ─────────────────────────────────────────────
#  TOOL FUNCTIONS
# ─────────────────────────────────────────────

def plan_project(prompt: str):
    print(f"🔨 Tool called: plan_project()")

    planner_prompt = """
    You are an expert software architect.
    Given a project description, return a JSON plan with this exact structure:
    {
      "project_name": "snake_case_name",
      "description": "one line summary",
      "tech_stack": ["list", "of", "technologies"],
      "files": [
        {
          "path": "relative/path/to/file.ext",
          "description": "what this file does"
        }
      ]
    }
    Rules:
    - Always include README.md and .gitignore
    - Include all necessary config files (package.json, requirements.txt, etc.)
    - Return STRICT JSON only, no markdown, no explanation
    """

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        config=types.GenerateContentConfig(
            system_instruction=planner_prompt,
            response_mime_type="application/json",
        ),
        contents=[{"role": "user", "parts": [{"text": prompt}]}],
    )

    return response.candidates[0].content.parts[0].text


def generate_file(input: str):
    print(f"🔨 Tool called: generate_file()")

    parsed = json.loads(input)
    file_path = parsed.get("path")
    file_description = parsed.get("description")
    project_name = parsed.get("project_name")
    tech_stack = parsed.get("tech_stack")
    project_description = parsed.get("project_description")

    code_prompt = f"""
    You are an expert developer. Write complete, clean, production-ready starter code.

    Project: {project_name}
    Tech Stack: {tech_stack}
    Project Description: {project_description}
    File: {file_path}
    File Purpose: {file_description}

    Rules:
    - Return ONLY the raw file content
    - No markdown fences, no explanations, no backticks
    - Write well-commented, functional code
    - For config files, include all realistic dependencies
    - For README.md, include setup and run instructions
    - For .gitignore, include all relevant patterns for the tech stack
    """

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[{"role": "user", "parts": [{"text": code_prompt}]}],
    )

    content = response.candidates[0].content.parts[0].text

    # Strip accidental markdown fences
    if content.strip().startswith("```"):
        lines = content.strip().split("\n")
        content = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

    return json.dumps({"path": file_path, "content": content})


def write_file(input: str):
    print(f"🔨 Tool called: write_file()")

    parsed = json.loads(input)
    relative_path = parsed.get("path")
    content = parsed.get("content")
    project_name = parsed.get("project_name")

    full_path = pathlib.Path("output") / project_name / relative_path
    full_path.parent.mkdir(parents=True, exist_ok=True)
    full_path.write_text(content, encoding="utf-8")

    return f"Success: {full_path} written ({len(content)} chars)"

available_tools = {
    "plan_project": {
        "function": plan_project,
        "description": "Takes a project prompt as input and returns a detailed JSON plan with project name, tech stack, and list of files to create"
    },
    "generate_file": {
        "function": generate_file,
        "description": "Takes a JSON string with path, description, project_name, tech_stack, project_description — returns file content as a JSON string"
    },
    "write_file": {
        "function": write_file,
        "description": "Takes a JSON string with path, content, project_name — writes the file to disk and returns success or failure"
    }
}

system_prompt = """
You are an expert AI agent that generates complete project boilerplate for any tech stack.
You work in start → plan → action → observe → output mode.

For any user request, you will:
1. Plan what files the project needs
2. Generate code for each file one by one
3. Write each file to disk one by one
4. Give a final summary when all files are done

Rules:
- Follow the output JSON format strictly
- Always perform ONE step at a time and wait for the next input
- Always call plan_project FIRST before generating any files
- After getting the plan, loop through every file: generate_file → write_file → next file
- Never skip any file from the plan
- Always use lowercase for the "step" field value
- The input field must always be a valid JSON string when calling generate_file or write_file

Output Format:
{
  "step": "string",
  "content": "string",
  "function": "function name — only include this when step is action",
  "input": "input for the function — only include this when step is action"
}

Available Tools:
- plan_project: Takes a project prompt as input and returns a detailed JSON plan with project name, tech stack, and list of files to create
- generate_file: Takes a JSON string with path, description, project_name, tech_stack, project_description — returns file content as a JSON string
- write_file: Takes a JSON string with path, content, project_name — writes the file to disk and returns success or failure

Example:

User Query: I want to build a REST API in Node.js

Output: {"step": "plan", "content": "The user wants a Node.js REST API. I will first call plan_project to get the file structure."}
Output: {"step": "action", "function": "plan_project", "input": "Build a REST API in Node.js with Express"}
Output: {"step": "observe", "content": "Got the plan back. Project has 5 files: server.js, routes/index.js, package.json, .env.example, README.md"}
Output: {"step": "plan", "content": "Now I will generate and write each file one by one. Starting with server.js"}
Output: {"step": "action", "function": "generate_file", "input": "{\"path\": \"server.js\", \"description\": \"Express server entry point\", \"project_name\": \"node_rest_api\", \"tech_stack\": \"Node.js, Express\", \"project_description\": \"A REST API built with Node.js and Express\"}"}
Output: {"step": "observe", "content": "server.js code generated successfully"}
Output: {"step": "action", "function": "write_file", "input": "{\"path\": \"server.js\", \"content\": \"...generated content...\", \"project_name\": \"node_rest_api\"}"}
Output: {"step": "observe", "content": "server.js written to disk successfully"}
Output: {"step": "plan", "content": "Moving to next file: routes/index.js"}
... repeat for every file ...
Output: {"step": "output", "content": "Your Node.js REST API is ready! 5 files created in output/node_rest_api/. Run: cd output/node_rest_api && npm install && node server.js"}
"""

def execute_tool(tool_name: str, tool_input: str):
    if tool_name in available_tools:
        return available_tools[tool_name]["function"](tool_input)
    return f"Error: tool '{tool_name}' not found"


def run_agent():
    contents = []

    while True:
        user_input = input("\n📝 What do you want to build?\n> ").strip()
        if not user_input:
            continue

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
                contents=contents,
            )

            parsed_output = json.loads(response.candidates[0].content.parts[0].text)

            contents.append({
                "role": "model",
                "parts": [{"text": json.dumps(parsed_output)}]
            })

            if parsed_output.get("step") == "plan":
                print(f"🧠 {parsed_output.get('content')}")
                continue

            if parsed_output.get("step") == "action":
                tool_name  = parsed_output.get("function")
                tool_input = parsed_output.get("input")
                print(f"🔧 Calling: {tool_name}")

                tool_output = execute_tool(tool_name, tool_input)

                observation = json.dumps({"step": "observation", "output": tool_output})
                contents.append({
                    "role": "user",
                    "parts": [{"text": observation}]
                })
                continue

            if parsed_output.get("step") == "output":
                print(f"\n🤖 {parsed_output.get('content')}")
                break

if __name__ == "__main__":
    print("🤖 AI Boilerplate Agent — Ready")
    print("─" * 40)
    run_agent()