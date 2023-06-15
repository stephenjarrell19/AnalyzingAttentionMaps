import yaml
import sys
import re
import pdb
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
    return tokenizer([re.split(r'(\s)', x) for x in examples["text"]], truncation=True, max_length=2048)


if __name__ == '__main__':

    # Get Training Config
    with open(f'options/{sys.argv[1]}.yaml','r') as f:
        cfg = yaml.safe_load(f)

    # Get Dataset
    if cfg["Dataset"] == 'arxiv':
        dataset = load_arxiv_dataset(cfg)

    dataset = dataset.train_test_split(test_size=0.2)
    pdb.set_trace()
    tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-neo-125M')
    tokenized_dataset = dataset.map(
        preprocess_function,
        batched=True,
        num_proc=4,
    )

    tokenizer.pad_token = tokenizer.eos_token
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    lm_dataset = tokenized_dataset.map(group_texts, batched=True, num_proc=4)
    # Get Model
    model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-125M")

    # Train Model
    training_args = TrainingArguments(
        output_dir="arxiv-model",
        evaluation_strategy="epoch",
        learning_rate=1e-3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4, 
        fp16=True,
        num_train_epochs=4,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model(cfg["save_path"]+"gpt-neo")
    tokenizer.save_pretrained(cfg["save_path"]+"tokenizer")

    # Save Model
