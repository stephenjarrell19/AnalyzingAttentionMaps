import json
from datasets import Dataset

def load_arxiv(data_path: str="data/", size: int=10000):

    try:
        arxiv = []
        with open(data_path, 'r') as file:
            # 2272690 Lines
            for i, line in enumerate(file):
                if i > size:
                    break
                arxiv.append(line)
    except:
        raise ValueError(f'Data file not found at path {data_path}')

    return arxiv

def preprocess_arxiv(data):

    relevant_keys = ['title', 'abstract']
    for i, line in enumerate(data):
        line = json.loads(line)

        remove_keys = set(line.keys()) - set(relevant_keys)
        for k in remove_keys:
            del line[k] 

        for k,v in line.items():
            v = v.replace("\n", " ")
            v = v.strip()
            line[k] = v
        
        data[i] = line
    
    arxiv = {'text': []}
    for i, item in enumerate(data):
        value = item['title'] + ': ' +item['abstract']
        # Sequence limit of GPT-Neo125M
        if len(value) <= 2048:
            arxiv['text'].append(value)
    return arxiv

def load_arxiv_dataset(cfg):
    data_path = cfg["data_path"]
    dataset_size = cfg["size"]
    data = load_arxiv(data_path)
    data = preprocess_arxiv(data)
    dataset = Dataset.from_dict(data)

    return dataset