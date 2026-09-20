#!/usr/bin/env python3

import pandas as pd
clean_text    = __import__('1-clean_text').clean_text
tokenize_text = __import__('2-tokenize').tokenize_text


# tweet vs word vs split
msg = clean_text("Don't call me!!! Visit <URL> for FREE prizes... u won :)")
print(f"input : {msg}\n")

for method in ['tweet', 'word', 'split']:
    tokens = tokenize_text(msg, method=method)
    print(f"[{method}] ({len(tokens)}) {tokens}\n")