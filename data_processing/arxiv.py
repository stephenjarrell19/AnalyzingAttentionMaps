import json
from datasets import Dataset

def load_arxiv(data_path: str="data/"):

    try:
        arxiv = []
        with open(data_path, 'r') as file:
            # 2272690 Lines
            for i, line in enumerate(file):
                if i > 1000000:
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
        arxiv['text'].append(item['title'] + ': ' +item['abstract'])
    return arxiv

def load_arxiv(data_path="../data/arxiv_dataset.json"):

    data = load_arxiv(data_path="../data/arxiv_dataset.json")
    data = preprocess_arxiv(data)
    dataset = Dataset.from_dict(data)

    return dataset