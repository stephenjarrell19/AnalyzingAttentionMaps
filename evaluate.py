import yaml
import sys
from bertviz import model_view
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer
from transformers import DataCollatorForLanguageModeling
from data_processing.arxiv import *
from data_processing.common_crawl import *
from data_processing.poems import *


def group_texts(examples):
    block_size = 128
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])

    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size
    # Split by chunks of block_size.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result


def preprocess_function(examples):
    return tokenizer([" ".join(x) for x in examples["text"]], truncation=True, max_length=2048)


if __name__ == '__main__':

    # Get Training Config
    with open(f'options/{sys.argv[1]}.yaml','r') as f:
        cfg = yaml.safe_load(f)

    prompt = "One day"

    # Get Dataset
    if cfg["Dataset"] == 'arxiv':
        dataset = load_arxiv_dataset(cfg)
        model = AutoModelForCausalLM.from_pretrained(cfg['save_path']+"/fine_tuned_model/", output_attentions=True)

    dataset = dataset.train_test_split(test_size=0.2)
    tokenizer = AutoTokenizer.from_pretrained(cfg['save_path']+"/tokenizer")
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    outputs = model.generate(inputs, max_new_tokens=100, do_sample=True, top_k=50, top_p=0.95)
    output_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    tokens = tokenizer.convert_ids_to_tokens(inputs[0])
    attention = model(inputs)[-1]
    torch.save(torch.stack(attention), 'arxiv.pt')
    model_view(attention, tokens)
