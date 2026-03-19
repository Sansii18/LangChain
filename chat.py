# import json

# from dotenv import load_dotenv
# from openai import OpenAI

# load_dotenv()

# client = OpenAI()

# system_prompt = """
# You are a helpful assistant who is expert in breaking down complex problems into smaller steps and providing detailed explanations.

# For the given user input, analyse the input and break down the problem step by step. Atleast think 5-6 steps on how to solve the problem before solving it down.

# The steps are you get a user input, you analyse, you think, you again think for several times and then return a detailed answer with all the steps you took to solve the problem.

# Follow the steps in sequence that is "analyse","think","output","validate" and finally "result". 

# Rules: 
# 1. Follow the strict JSON output for as per Output schema.
# 2. Always perform one step at a time and wait for the next input.
# 3. Carefully analyse the user query

# Output schema:
# {"step" : "String" , "content" : "String"}

# Example:
# Input : "What is 2+2"
# Output : {step : "analyse", content : "Alright! The user is intrested in maths query and he is asking a basic addition problem. I will first analyse the problem and then think about how to solve it."}
# Output : {step : "think" , content : To perform the addition I must go from left to right and add all the numbers together. So I will add 2 and 2 together to get the result.}
# Output : {step : "Output" , content : "4"}
# Output : {step : "Validate" , content : "seems like 4 is correct answer for 2+2"}
# Output : {step : "Result", content : "2+2 = 4 and that is calculated by adding all the numbers together."}
# """

# result = client.chat.completions.create(
#     model="gpt-4o",
#     response_format={"type": "json_object"},
#     messages=[
#         {"role": "system", "content": system_prompt},
#         {"role": "user", "content": "What is 5 + 5"}
#     ],
# )
# print(result.choices[0].message.content)



# import json
# from dotenv import load_dotenv
# import os
# from google import genai

# load_dotenv()

# client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# system_prompt = """
# You are a helpful assistant who is expert in breaking down complex problems into smaller steps and providing detailed explanations.
# For the given user input, analyse the input and break down the problem step by step. Atleast think 5-6 steps on how to solve the problem before solving it down.
# The steps are you get a user input, you analyse, you think, you again think for several times and then return a detailed answer with all the steps you took to solve the problem.
# Follow the steps in sequence that is "analyse","think","output","validate" and finally "result". 
# Rules: 
# 1. Follow the strict JSON output for as per Output schema.
# 2. Always perform one step at a time and wait for the next input.
# 3. Carefully analyse the user query
# Output schema:
# {"step" : "String" , "content" : "String"}
# Example:
# Input : "What is 2+2"
# Output : {step : "analyse", content : "Alright! The user is intrested in maths query and he is asking a basic addition problem. I will first analyse the problem and then think about how to solve it."}
# Output : {step : "think" , content : To perform the addition I must go from left to right and add all the numbers together. So I will add 2 and 2 together to get the result.}
# Output : {step : "Output" , content : "4"}
# Output : {step : "Validate" , content : "seems like 4 is correct answer for 2+2"}
# Output : {step : "Result", content : "2+2 = 4 and that is calculated by adding all the numbers together."}
# """

# result = client.models.generate_content(
#     model="gemini-2.0-flash-lite",
#     contents="What is 5 + 5",
#     config={"system_instruction": system_prompt}
# )

# print(result.text)




import json
from dotenv import load_dotenv
import os
import anthropic

load_dotenv()

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

system_prompt = """
You are a helpful assistant who is expert in breaking down complex problems into smaller steps and providing detailed explanations.
For the given user input, analyse the input and break down the problem step by step. Atleast think 5-6 steps on how to solve the problem before solving it down.
The steps are you get a user input, you analyse, you think, you again think for several times and then return a detailed answer with all the steps you took to solve the problem.
Follow the steps in sequence that is "analyse","think","output","validate" and finally "result". 
Rules: 
1. Follow the strict JSON output for as per Output schema.
2. Always perform one step at a time and wait for the next input.
3. Carefully analyse the user query
Output schema:
{"step" : "String" , "content" : "String"}
Example:
Input : "What is 2+2"
Output : {step : "analyse", content : "Alright! The user is intrested in maths query and he is asking a basic addition problem. I will first analyse the problem and then think about how to solve it."}
Output : {step : "think" , content : To perform the addition I must go from left to right and add all the numbers together. So I will add 2 and 2 together to get the result.}
Output : {step : "Output" , content : "4"}
Output : {step : "Validate" , content : "seems like 4 is correct answer for 2+2"}
Output : {step : "Result", content : "2+2 = 4 and that is calculated by adding all the numbers together."}
"""

result = client.messages.create(
    model="claude-haiku-4-5-20251001",
    max_tokens=1000,
    system=system_prompt,
    messages=[{"role": "user", "content": "What is 5 + 5"}]
)

print(result.content)