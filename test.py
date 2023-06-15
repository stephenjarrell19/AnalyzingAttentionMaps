prompt = "The cat sat on the mat"
from transformers import pipeline

generator = pipeline("text-generation", model="roy23roy/my_poem_model_final",max_length=100)
generator(prompt)

#Attention map
from transformers import AutoTokenizer, AutoModel, utils
from bertviz import model_view
utils.logging.set_verbosity_error()  

model_name = "roy23roy/my_poem_model_final" 
input_text = "The cat sat on the mat"
model = AutoModel.from_pretrained(model_name, output_attentions=True)  
tokenizer = AutoTokenizer.from_pretrained(model_name)
inputs = tokenizer.encode(input_text, return_tensors='pt')  
outputs = model(inputs)  
attention = outputs[-1]  
tokens = tokenizer.convert_ids_to_tokens(inputs[0])  
model_view(attention, tokens)  

import torch
torch.save(torch.stack(attention),'poems.pt')
