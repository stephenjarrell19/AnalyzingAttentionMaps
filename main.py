import yaml
import sys
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer

from data_processing.arxiv import *
from data_processing.common_crawl import *
from data_processing.poems import *


def preprocess_function(examples):
    return tokenizer([" ".join(x) for x in examples["text"]], truncation=True, max_length=2048)


if __name__ == '__main__':

    # Get Training Config
    with open(f'options/{sys.argv[1]}.yaml','r') as f:
        cfg = yaml.safe_load(f)

    # Get Dataset
    if cfg["Dataset"] == 'arxiv':
        dataset = load_arxiv_dataset(cfg)

    dataset = dataset.train_test_split(test_size=0.2)
    tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-neo-125M')
    tokenized_dataset = dataset.map(
        preprocess_function,
        batched=True,
        num_proc=4,
    )

    # Get Model
    model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-125M")

    # Train Model


    # Save Model
