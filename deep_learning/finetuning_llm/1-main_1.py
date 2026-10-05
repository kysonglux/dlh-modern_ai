#!/usr/bin/env python3

load_emotion_dataset = __import__('0-load_dataset').load_emotion_dataset
load_distilbert = __import__('1-load_distilbert').load_distilbert

dataset = load_emotion_dataset()

model_name = "distilbert-base-uncased"
num_classes = 6
id_to_label = {0: "sadness", 1: "joy", 2: "love", 3: "anger", 4: "fear", 5: "surprise"}
label_to_id = {v: k for k, v in id_to_label.items()}

tokenizer, _ = load_distilbert(model_name, num_classes, id_to_label, label_to_id)

print("Tokenizer type:", type(tokenizer))
print("\nTokenizer Vocabulary Size:", tokenizer.vocab_size)
print("Tokenizer Max Input Length:", tokenizer.model_max_length)
print("Tokenizer Model Input Names:", tokenizer.model_input_names)