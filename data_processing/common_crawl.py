from transformers import DataCollatorForLanguageModeling
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer
from transformers import AutoTokenizer, GPTNeoModel
import torch
from transformers import AutoTokenizer, AutoModel, utils
from bertviz import model_view

def get_attention_crawl(intput):
  utils.logging.set_verbosity_error()  # Suppress standard warnings
  tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
  model = AutoModelForCausalLM.from_pretrained("Model_Crawl",local_files_only=True)
  inputs = tokenizer.encode(input,return_tensors="pt")
  out = model(inputs,output_attentions=True)
  attention = out[-1]
  tokens = tokenizer.convert_ids_to_tokens(inputs[0])  # Convert input ids to token strings
  model_view(attention, tokens)  # Display model view
  return attention
  
