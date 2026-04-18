from operator import add
from pyexpat.errors import messages

from mem0 import Memory
import os
from openai import OpenAI

NEO4J_URL = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

config = {
    "version": "v1.1",
    "embedder" : {
        "provider": "openai",
        "config": {"api_key": os.getenv("OPENAI_API_KEY") , 
                   "model": "text-embedding-3-small"}
    },

    "llm": {
        "provider": "openai",
        "config": {"api_key": os.getenv("OPENAI_API_KEY") ,
                   "model": "gpt-4.1"}
    },

    "vector_store": {
        "provider": "qdrant",
        "config": {
            "host" : QUADRANT_HOST,
            "port" : 6333,
        }
    },

    "graph_db": {
        "provider": "neo4j",
        "config": {
            "url": os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            "username": os.getenv("NEO4J_USERNAME", "neo4j"),
            "password": os.getenv("NEO4J_PASSWORD", "password")
        }
    }
}

mem_client = Memory.from_config(config)
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def chat(message):
    mem_result = mem_client.search(message, user_id = "p123")  # for now this id is hardcoded, in a real application it should be dynamic (use id of a database record)

    print("Memory search result : ", mem_result.get("results"))

    memories = "\n".join([m["memory"] for m in mem_result])

    print("Memories : ", memories)

    SYSTEM_PROMPT = """
    You are a Memory - Aware Fact Extraction Agent , an advanced AI designed to systematically analyze input content, extract structured knowledge and maintain in optimized memory store. Your primary function is information distillation and knowledge preservation with contextual awarness. 

    Tone : Professional analytical , precise - focused , with clear uncertainty signaling

    {memories}
    """

    message = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": message}
    ]

    response = openai_client.chat.completions.create(
        model="gpt-4.1",
        messages=message
    )

    messages.append(
        "role": "assistant",
        "content": response.choices[0].message.content
    )

    mem_client.add(messages, user_id = "p123")  # for now this id is hardcoded, in a real application it should be dynamic (use id of a database record)

    return response.choices[0].message.content

while True:
    message = input(">> ")
    print("BOT : ", chat(message))

