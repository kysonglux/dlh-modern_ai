#!/usr/bin/env python3

load_emotion_dataset = __import__('0-load_dataset').load_emotion_dataset
load_distilbert = __import__('1-load_distilbert').load_distilbert
tokenize_and_map = __import__('2-tokenize_and_map').tokenize_and_map
create_data_collator = __import__('3-dynamic_padding').create_data_collator

from transformers import logging
logging.set_verbosity_error()

dataset = load_emotion_dataset()

label_names = ["sadness", "joy", "love", "anger", "fear", "surprise"]
label_to_id = {name: idx for idx, name in enumerate(label_names)}
id_to_label = {idx: name for idx, name in enumerate(label_names)}
num_classes = len(label_names)
model_name = "distilbert-base-uncased"

tokenizer, _ = load_distilbert(model_name, num_classes, id_to_label, label_to_id)

tokenized_train, tokenized_val, tokenized_test = tokenize_and_map(
    dataset,
    tokenizer,
    max_length=128,
    truncation=True,
    batched=True
)

data_collator = create_data_collator(tokenizer)

def get_batch(start_idx, batch_size=4):
  return [
      {k: tokenized_train[i][k] for k in ["input_ids", "attention_mask"]}
      for i in range(start_idx, start_idx + batch_size)
  ]

batch1_examples = get_batch(0)
batch2_examples = get_batch(4)

batch1 = data_collator(batch1_examples)
batch2 = data_collator(batch2_examples)


for i, batch in enumerate([batch1, batch2], 1):
  print(f"Batch {i} shapes:")
  print("  input_ids:", batch['input_ids'].shape)
  print("  Input IDs:")
  print(batch['input_ids'])
  print("  attention_mask:", batch['attention_mask'].shape)
  print()

  ## Decode token IDs back to text ##
  print(f"  Decoded texts from Batch {i}:")
  for j, ids in enumerate(batch['input_ids']):
    decoded_text = tokenizer.decode(ids, skip_special_tokens=True)
    print(f"    Example {j+1}: {decoded_text}")
  print()

  ## Show with special tokens ##
  print(f"  Decoded texts (with special tokens):")
  for j, ids in enumerate(batch['input_ids']):
    decoded_text = tokenizer.decode(ids, skip_special_tokens=False)
    print(f"    Example {j+1}: {decoded_text}")
  print()