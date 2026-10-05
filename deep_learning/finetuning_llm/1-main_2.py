#!/usr/bin/env python3

load_emotion_dataset = __import__('0-load_dataset').load_emotion_dataset
load_distilbert = __import__('1-load_distilbert').load_distilbert

dataset = load_emotion_dataset()

model_name = "distilbert-base-uncased"
num_classes = 6
id_to_label = {0: "sadness", 1: "joy", 2: "love", 3: "anger", 4: "fear", 5: "surprise"}
label_to_id = {v: k for k, v in id_to_label.items()}

_, model = load_distilbert(model_name, num_classes, id_to_label, label_to_id)

print("Model type:", type(model))
print("\nModel Configuration:")
print(model.config)


total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nTotal parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
