from fastapi import FastAPI
from ollama import Client 
from fastapi import Body

app = FastAPI()
client = Client(
    host = "http://localhost:11434"
)

client.pull("gemma3:1b")

@app.get("/chat")
def chat(messages: str = Body(..., description= "Chat Messages")):
    response = client.chat(model = "gemma3:1b", messages = [
        {"role": "user", "content" : messages}
    ])

    return response['messages']['content']