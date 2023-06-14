import yaml
import sys

from data_processing.arxiv import *
from data_processing.common_crawl import *
from data_processing.poems import *







if __name__ == '__main__':

    # Get Training Config
    with open(f'options/{sys.argv[1]}.yaml','r') as f:
        cfg = yaml.safe_load(f)

    # Get Data
    if cfg["Dataset"] == 'arxiv':
        data = load_arxiv()

    # Get Model


    # Train Model

    # Save Model
