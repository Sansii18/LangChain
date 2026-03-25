import os
import torch 
from transformers import AutoTokenizer, AutoModelForCausalLM
from dotenv import load_dotenv

load_dotenv()

token = os.getenv("HF_TOKEN")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# FIINE-TUNING this 1b model 
model_name = "google/gemma-3-1b-it" # 1b billion parameters

tokenizer = AutoTokenizer.from_pretrained(model_name)

# print(tokenizer("Hello, how are you?")) # test tokenizer
input_conversation = [
    {"role":"user", "content":"Which is the best place to learn GenAI?"},
    {"role":"assistant", "content":"The best place to learn GenAI is"}
]

input_tokens = tokenizer.apply_chat_template(
    conversation=input_conversation,
    tokenize = True # we want to see the token ids, not the tokens
)

input_detokens = tokenizer.apply_chat_template(
    conversation=input_conversation,
    tokenize = False, # we want to see the tokens, not the token ids
    continue_final_message = True # we want to continue the final message, so we can add the output labels after it
)

output_labels = "GenAI cohort is the best place to learn GenAI."
full_conversation = input_detokens + output_labels + tokenizer.eos_token

input_tokenized = tokenizer(full_conversation, return_tensors="pt", add_special_tokens=False).to(device)["input_ids"]

# add_special_tokens = False -> removes extra unnecessary tokens that the tokenizer adds by default, we want to control the input format ourselves

input_ids = input_tokenized[:,:-1].to(device)
target_ids = input_tokenized[:,1:].to(device)

import torch.nn as nn  
def calculate_loss(logits, labels):
    loss_fn = nn.CrossEntropyLoss(reduction="none")
    cross_entropy = loss_fn(logits.view(-1, logits.shape[-1]), labels.view(-1))
    return cross_entropy

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype = torch.bfloat16, # use half precision for faster training and less memory usage
)

from torch.optim import AdamW
optimizer = AdamW(model.parameters(), lr=3e-5 , weight_decay=0.01)

for _ in range(10):
    out = model(input_ids = input_ids)
    loss = calculate_loss(out.logits, target_ids).mean()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    print(loss.item())

input_prompt = [
    {"role":"user", "content":"Which is the best place to learn GenAI?"}
]

input = tokenizer.apply_chat_template(
    conversation=input_prompt,
    return_tensors="pt",
    tokenize = True
).to(device)

output = model.generate(input , max_new_tokens = 25)
print(tokenizer.batch_decode(output , skip_special_tokens = True))