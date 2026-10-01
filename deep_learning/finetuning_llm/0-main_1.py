#!/usr/bin/env python3

load_emotion_dataset = __import__('0-load_dataset').load_emotion_dataset

dataset = load_emotion_dataset()

print(type(dataset))

print("\nDataset Structure:")
print(dataset)

print("\nDataset Column Names:")
print(dataset.column_names)

print("\nDataset Shape:")
print(dataset.shape)