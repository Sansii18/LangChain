import os 
from dotenv import load_dotenv
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

load_dotenv()

token = os.getenv("HF_TOKEN")
model_name = "google/gemma-3-1b-it"

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    token=token
)

print(tokenizer("What is the weather of New York?"))

input_tokens = tokenizer(
    "What is the weather of New York?",
    return_tensors="pt"
)["input_ids"]

print(input_tokens) 

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    token=token,
    torch_dtype=torch.float32   # safer for Mac
)

gen_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer
)

output = gen_pipeline(
    "What is the weather of New York?",
    max_length=100
)

print(output)