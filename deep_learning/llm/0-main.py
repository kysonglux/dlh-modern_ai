#!/usr/bin/env python3

load_mlm = __import__('0-load_mlm').load_mlm

model = load_mlm("roberta-base")
print("Model type:", type(model))