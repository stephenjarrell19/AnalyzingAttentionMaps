def preprocess_function(examples):
    return tokenizer(["".join(x) for x in examples["poem"]])
  
block_size = 128


def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size
    # Split by chunks of block_size.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

from transformers import TrainingArguments
!pip install accelerate -U
training_args = TrainingArguments(
    output_dir="my_poem_model_final",
    evaluation_strategy = "epoch",
    learning_rate=1e-4,
    weight_decay=0.01,
    push_to_hub=True,
    num_train_epochs=10,
    per_device_train_batch_size=8,  # batch size per device during training
    per_device_eval_batch_size=16,
    

)

