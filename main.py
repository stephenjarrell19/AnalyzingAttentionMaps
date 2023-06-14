import yaml
import sys
from transformers import AutoTokenizer
from data_processing.arxiv import *
from data_processing.common_crawl import *
from data_processing.poems import *


if __name__ == '__main__':

    # Get Training Config
    with open(f'options/{sys.argv[1]}.yaml','r') as f:
        cfg = yaml.safe_load(f)

    # Get Dataset
    if cfg["Dataset"] == 'arxiv':
        dataset = load_arxiv_dataset(cfg["data_path"])

    # Get Model
    dataset = dataset.train_test_split(test_size=0.2)
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")

    print(dataset['train'][0])
    # Train Model

    # Save Model
